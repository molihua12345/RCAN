#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Comparison Script for RCAN vs RCAN_CBAM

This script compares the performance of two super-resolution models:
1. RCAN_x4 (original model)
2. RCAN_CBAM_x4 (improved model with CBAM attention)

The script evaluates multiple metrics and generates comprehensive visualizations.
"""

import os
import time
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.feature import local_binary_pattern
from scipy import ndimage

import model
from dataset import TestImageDataset, CUDAPrefetcher
from imgproc import tensor_to_image
from utils import build_iqa_model, load_state_dict, make_directory, AverageMeter
from config_comparison import MODEL_CONFIGS, TEST_DATASET_CONFIG, TEST_CONFIG, OUTPUT_CONFIG, VIZ_CONFIG, METRICS_CONFIG

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class ModelComparator:
    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self.results = {
            'rcan_x4': {'metrics': [], 'images': []},
            'rcan_cbam_x4': {'metrics': [], 'images': []}
        }
        
        # Load configurations
        self.model_configs = MODEL_CONFIGS
        self.test_config = TEST_CONFIG
        self.dataset_config = TEST_DATASET_CONFIG
        self.output_config = OUTPUT_CONFIG
        
        # Results directory
        self.results_dir = self.output_config['results_dir']
        make_directory(self.results_dir)
        make_directory(os.path.join(self.results_dir, self.output_config['images_subdir']))
        make_directory(os.path.join(self.results_dir, self.output_config['plots_subdir']))
        make_directory(os.path.join(self.results_dir, self.output_config['data_subdir']))
        
    def load_dataset(self) -> CUDAPrefetcher:
        """Load test dataset"""
        # Use configuration for dataset paths
        base_dir = self.dataset_config['base_dir']
        gt_dir = os.path.join(base_dir, self.dataset_config['gt_subdir'])
        lr_dir = os.path.join(base_dir, self.dataset_config['lr_subdir'])
        
        test_datasets = TestImageDataset(gt_dir, lr_dir)
        test_dataloader = DataLoader(test_datasets,
                                   batch_size=self.test_config['batch_size'],
                                   shuffle=False,
                                   num_workers=self.test_config['num_workers'],
                                   pin_memory=True,
                                   drop_last=False,
                                   persistent_workers=False)
        test_data_prefetcher = CUDAPrefetcher(test_dataloader, self.device)
        return test_data_prefetcher
    
    def load_model(self, model_key: str) -> nn.Module:
        """Load and return a model"""
        config = self.model_configs[model_key]
        sr_model = model.__dict__[config['arch']]()
        sr_model = sr_model.to(device=self.device)
        sr_model = load_state_dict(sr_model, config['weights_path'])
        sr_model.eval()
        return sr_model
    
    def calculate_advanced_metrics(self, sr_img: np.ndarray, gt_img: np.ndarray) -> Dict[str, float]:
        """Calculate advanced image quality metrics"""
        # Convert to grayscale for some metrics
        sr_gray = cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY) if len(sr_img.shape) == 3 else sr_img
        gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY) if len(gt_img.shape) == 3 else gt_img
        
        metrics = {}
        
        # Basic metrics
        metrics['psnr'] = psnr_skimage(gt_img, sr_img, data_range=255)
        metrics['ssim'] = ssim_skimage(gt_img, sr_img, data_range=255, channel_axis=2 if len(sr_img.shape) == 3 else None)
        
        # Mean Squared Error
        metrics['mse'] = np.mean((gt_img.astype(float) - sr_img.astype(float)) ** 2)
        
        # Mean Absolute Error
        metrics['mae'] = np.mean(np.abs(gt_img.astype(float) - sr_img.astype(float)))
        
        # Edge Preservation Index
        gt_edges = cv2.Canny(gt_gray, 100, 200)
        sr_edges = cv2.Canny(sr_gray, 100, 200)
        edge_similarity = np.sum(gt_edges & sr_edges) / (np.sum(gt_edges | sr_edges) + 1e-8)
        metrics['edge_preservation'] = edge_similarity
        
        # Gradient Magnitude Similarity
        gt_grad_x = cv2.Sobel(gt_gray, cv2.CV_64F, 1, 0, ksize=3)
        gt_grad_y = cv2.Sobel(gt_gray, cv2.CV_64F, 0, 1, ksize=3)
        sr_grad_x = cv2.Sobel(sr_gray, cv2.CV_64F, 1, 0, ksize=3)
        sr_grad_y = cv2.Sobel(sr_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gt_grad_mag = np.sqrt(gt_grad_x**2 + gt_grad_y**2)
        sr_grad_mag = np.sqrt(sr_grad_x**2 + sr_grad_y**2)
        
        grad_similarity = ssim_skimage(gt_grad_mag, sr_grad_mag, data_range=gt_grad_mag.max())
        metrics['gradient_similarity'] = grad_similarity
        
        # Texture Similarity (using Local Binary Pattern)
        radius = 3
        n_points = 8 * radius
        gt_lbp = local_binary_pattern(gt_gray, n_points, radius, method='uniform')
        sr_lbp = local_binary_pattern(sr_gray, n_points, radius, method='uniform')
        
        # Calculate histogram correlation
        gt_hist, _ = np.histogram(gt_lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        sr_hist, _ = np.histogram(sr_lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        
        gt_hist = gt_hist.astype(float)
        sr_hist = sr_hist.astype(float)
        gt_hist /= (gt_hist.sum() + 1e-8)
        sr_hist /= (sr_hist.sum() + 1e-8)
        
        texture_similarity = np.corrcoef(gt_hist, sr_hist)[0, 1]
        metrics['texture_similarity'] = texture_similarity if not np.isnan(texture_similarity) else 0.0
        
        # Sharpness metric (variance of Laplacian)
        gt_laplacian = cv2.Laplacian(gt_gray, cv2.CV_64F)
        sr_laplacian = cv2.Laplacian(sr_gray, cv2.CV_64F)
        
        gt_sharpness = gt_laplacian.var()
        sr_sharpness = sr_laplacian.var()
        
        # Sharpness preservation ratio
        metrics['sharpness_ratio'] = sr_sharpness / (gt_sharpness + 1e-8)
        
        return metrics
    
    def test_model(self, model_key: str, model: nn.Module, data_prefetcher: CUDAPrefetcher) -> List[Dict]:
        """Test a single model and return results"""
        print(f"\nTesting {self.model_configs[model_key]['display_name']}...")
        
        # Build IQA models
        psnr_model, ssim_model = build_iqa_model(4, True, self.device)
        
        results = []
        batch_index = 0
        
        # Reset data prefetcher
        data_prefetcher.reset()
        batch_data = data_prefetcher.next()
        
        while batch_data is not None and batch_index < self.test_config['max_test_images']:
            # Load data
            gt = batch_data["gt"].to(device=self.device, non_blocking=True)
            lr = batch_data["lr"].to(device=self.device, non_blocking=True)
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                sr = model(lr)
            inference_time = time.time() - start_time
            
            # Convert tensors to images
            sr_image = tensor_to_image(sr, range_norm=False, half=False)
            gt_image = tensor_to_image(gt, range_norm=False, half=False)
            lr_image = tensor_to_image(lr, range_norm=False, half=False)
            
            # Calculate PyTorch-based metrics
            psnr_torch = psnr_model(sr, gt).item()
            ssim_torch = ssim_model(sr, gt).item()
            
            # Calculate advanced metrics
            advanced_metrics = self.calculate_advanced_metrics(sr_image, gt_image)
            
            # Combine all metrics
            result = {
                'image_id': batch_index,
                'inference_time': inference_time,
                'psnr_torch': psnr_torch,
                'ssim_torch': ssim_torch,
                **advanced_metrics
            }
            
            results.append(result)
            
            # Save sample images for visualization
            if batch_index < self.test_config['save_sample_images']:
                self.save_comparison_image(lr_image, sr_image, gt_image, 
                                         model_key, batch_index, result)
            
            print(f"Image {batch_index + 1}: PSNR={psnr_torch:.2f}dB, SSIM={ssim_torch:.4f}, Time={inference_time:.3f}s")
            
            # Next batch
            batch_data = data_prefetcher.next()
            batch_index += 1
        
        return results
    
    def save_comparison_image(self, lr_img: np.ndarray, sr_img: np.ndarray, gt_img: np.ndarray,
                            model_key: str, img_id: int, metrics: Dict):
        """Save comparison images"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(lr_img)
        axes[0].set_title('Low Resolution')
        axes[0].axis('off')
        
        axes[1].imshow(sr_img)
        axes[1].set_title(f'{self.model_configs[model_key]["display_name"]}\nPSNR: {metrics["psnr"]:.2f}dB')
        axes[1].axis('off')
        
        axes[2].imshow(gt_img)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, self.output_config['images_subdir'], f'{model_key}_sample_{img_id:03d}.png'), 
                   dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def generate_comparison_plots(self):
        """Generate comprehensive comparison plots"""
        print("\nGenerating comparison plots...")
        
        # Convert results to DataFrames
        df_rcan = pd.DataFrame(self.results['rcan_x4']['metrics'])
        df_cbam = pd.DataFrame(self.results['rcan_cbam_x4']['metrics'])
        
        df_rcan['model'] = self.model_configs['rcan_x4']['display_name']
        df_cbam['model'] = self.model_configs['rcan_cbam_x4']['display_name']
        
        df_combined = pd.concat([df_rcan, df_cbam], ignore_index=True)
        
        # 1. Overall Performance Comparison
        self.plot_overall_comparison(df_combined)
        
        # 2. Metric Distribution Plots
        self.plot_metric_distributions(df_combined)
        
        # 3. Correlation Analysis
        self.plot_correlation_analysis(df_combined)
        
        # 4. Performance vs Image ID
        self.plot_performance_trends(df_combined)
        
        # 5. Radar Chart
        self.plot_radar_chart(df_rcan, df_cbam)
        
        # 6. Statistical Summary
        self.generate_statistical_summary(df_rcan, df_cbam)
    
    def plot_overall_comparison(self, df: pd.DataFrame):
        """Plot overall performance comparison"""
        metrics = ['psnr', 'ssim', 'edge_preservation', 'gradient_similarity', 'texture_similarity']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            sns.boxplot(data=df, x='model', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.upper().replace("_", " ")} Comparison')
            axes[i].tick_params(axis='x', rotation=45)
        
        # Inference time comparison
        sns.boxplot(data=df, x='model', y='inference_time', ax=axes[5])
        axes[5].set_title('Inference Time Comparison')
        axes[5].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, self.output_config['plots_subdir'], 'overall_comparison.png'), 
                   dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def plot_metric_distributions(self, df: pd.DataFrame):
        """Plot metric distributions"""
        metrics = ['psnr', 'ssim', 'mse', 'mae']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            for model in df['model'].unique():
                model_data = df[df['model'] == model][metric]
                axes[i].hist(model_data, alpha=0.7, label=model, bins=20)
            
            axes[i].set_title(f'{metric.upper()} Distribution')
            axes[i].set_xlabel(metric.upper())
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, self.output_config['plots_subdir'], 'metric_distributions.png'), 
                   dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def plot_correlation_analysis(self, df: pd.DataFrame):
        """Plot correlation analysis"""
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Metric Correlation Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, self.output_config['plots_subdir'], 'correlation_analysis.png'), 
                   dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def plot_performance_trends(self, df: pd.DataFrame):
        """Plot performance trends across images"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['psnr', 'ssim', 'edge_preservation', 'inference_time']
        
        for i, metric in enumerate(metrics):
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                axes[i].plot(model_data['image_id'], model_data[metric], 
                           marker='o', label=model, alpha=0.7)
            
            axes[i].set_title(f'{metric.upper().replace("_", " ")} vs Image ID')
            axes[i].set_xlabel('Image ID')
            axes[i].set_ylabel(metric.upper())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, self.output_config['plots_subdir'], 'performance_trends.png'), 
                   dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def plot_radar_chart(self, df_rcan: pd.DataFrame, df_cbam: pd.DataFrame):
        """Plot radar chart for model comparison"""
        metrics = ['psnr', 'ssim', 'edge_preservation', 'gradient_similarity', 'texture_similarity']
        
        # Calculate mean values
        rcan_means = [df_rcan[metric].mean() for metric in metrics]
        cbam_means = [df_cbam[metric].mean() for metric in metrics]
        
        # Normalize values to 0-1 scale for better visualization
        all_values = rcan_means + cbam_means
        min_val, max_val = min(all_values), max(all_values)
        
        rcan_norm = [(val - min_val) / (max_val - min_val) for val in rcan_means]
        cbam_norm = [(val - min_val) / (max_val - min_val) for val in cbam_means]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        rcan_norm += rcan_norm[:1]
        cbam_norm += cbam_norm[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, rcan_norm, 'o-', linewidth=2, label='RCAN (Original)', color='blue')
        ax.fill(angles, rcan_norm, alpha=0.25, color='blue')
        
        ax.plot(angles, cbam_norm, 'o-', linewidth=2, label='RCAN + CBAM (Improved)', color='red')
        ax.fill(angles, cbam_norm, alpha=0.25, color='red')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, self.output_config['plots_subdir'], 'radar_chart.png'), 
                   dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def generate_statistical_summary(self, df_rcan: pd.DataFrame, df_cbam: pd.DataFrame):
        """Generate statistical summary report"""
        metrics = ['psnr', 'ssim', 'mse', 'mae', 'edge_preservation', 
                  'gradient_similarity', 'texture_similarity', 'inference_time']
        
        summary_data = []
        
        for metric in metrics:
            rcan_mean = df_rcan[metric].mean()
            rcan_std = df_rcan[metric].std()
            cbam_mean = df_cbam[metric].mean()
            cbam_std = df_cbam[metric].std()
            
            improvement = ((cbam_mean - rcan_mean) / rcan_mean) * 100 if rcan_mean != 0 else 0
            
            summary_data.append({
                'Metric': metric.upper().replace('_', ' '),
                'RCAN Mean': f"{rcan_mean:.4f}",
                'RCAN Std': f"{rcan_std:.4f}",
                'CBAM Mean': f"{cbam_mean:.4f}", 
                'CBAM Std': f"{cbam_std:.4f}",
                'Improvement (%)': f"{improvement:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_df.to_csv(os.path.join(self.results_dir, self.output_config['data_subdir'], 'statistical_summary.csv'), index=False)
        
        # Create a nice table plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color code the improvement column
        for i in range(len(summary_df)):
            improvement_val = float(summary_df.iloc[i]['Improvement (%)'].replace('%', ''))
            if improvement_val > 0:
                table[(i+1, 5)].set_facecolor('#90EE90')  # Light green for positive
            elif improvement_val < 0:
                table[(i+1, 5)].set_facecolor('#FFB6C1')  # Light red for negative
        
        plt.title('Statistical Summary: RCAN vs RCAN+CBAM', size=16, pad=20)
        plt.savefig(os.path.join(self.results_dir, self.output_config['plots_subdir'], 'statistical_summary.png'), 
                   dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close()
        
        return summary_df
    
    def run_comparison(self):
        """Run the complete model comparison"""
        print("Starting Model Comparison...")
        print(f"Results will be saved to: {self.results_dir}")
        
        # Load dataset
        data_prefetcher = self.load_dataset()
        
        # Test both models
        for model_key in ['rcan_x4', 'rcan_cbam_x4']:
            try:
                model = self.load_model(model_key)
                results = self.test_model(model_key, model, data_prefetcher)
                self.results[model_key]['metrics'] = results
                
                # Calculate summary statistics
                avg_psnr = np.mean([r['psnr'] for r in results])
                avg_ssim = np.mean([r['ssim'] for r in results])
                avg_time = np.mean([r['inference_time'] for r in results])
                
                print(f"\n{self.model_configs[model_key]['display_name']} Summary:")
                print(f"  Average PSNR: {avg_psnr:.2f} dB")
                print(f"  Average SSIM: {avg_ssim:.4f}")
                print(f"  Average Inference Time: {avg_time:.3f} s")
                
            except Exception as e:
                print(f"Error testing {model_key}: {str(e)}")
                continue
        
        # Generate comparison plots
        if self.results['rcan_x4']['metrics'] and self.results['rcan_cbam_x4']['metrics']:
            self.generate_comparison_plots()
            
            # Save results to JSON
            with open(os.path.join(self.results_dir, self.output_config['data_subdir'], 'comparison_results.json'), 'w') as f:
                json.dump(self.results, f, indent=2)
            
            print(f"\nComparison complete! Results saved to {self.results_dir}")
            print("Generated files:")
            print(f"  - {self.output_config['data_subdir']}/comparison_results.json: Raw results data")
            print(f"  - {self.output_config['data_subdir']}/statistical_summary.csv: Statistical summary table")
            print(f"  - {self.output_config['plots_subdir']}/: All visualization plots")
            print(f"  - {self.output_config['images_subdir']}/: Sample comparison images")
        else:
            print("\nError: Could not test both models. Please check model paths and configurations.")


def main():
    """Main function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create comparator and run comparison
    comparator = ModelComparator(device)
    comparator.run_comparison()


if __name__ == "__main__":
    main()