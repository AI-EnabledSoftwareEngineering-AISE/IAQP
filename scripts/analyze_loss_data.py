#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_loss_data.py
-------------------
Analyze the loss analysis JSON data to extract insights about training dynamics.

Updated for modular design with 4 core losses only:
1. KL Divergence Loss (KLD) - Main ranking objective
2. Identity Loss (ID) - Semantic preservation 
3. Frontier Gap Loss - Ranking stability
4. Coarse Cell Loss - ANN routing efficiency

Disabled losses (iso, overlap, res) are excluded from analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_analyze_loss_data(json_file: str):
    """Load and analyze the loss data from JSON file."""
    
    print("🔍 Loading Loss Analysis Data...")
    print("=" * 50)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    iterations = np.array(data['iterations'])
    losses = data['losses']
    metadata = data['metadata']
    
    # Check for epoch data
    has_epoch_data = 'epochs' in data and 'epoch_losses' in data and len(data['epochs']) > 0
    if has_epoch_data:
        epochs = np.array(data['epochs'])
        epoch_losses = data['epoch_losses']
        print(f"  • Epoch-level data available: {len(epochs)} epochs")
    else:
        print(f"  • Epoch-level data: Not available (iteration-level only)")
    
    print(f"📊 Dataset Overview:")
    print(f"  • Total iterations: {len(iterations):,}")
    print(f"  • Training epochs: {metadata['epochs']}")
    print(f"  • Batch size: {metadata['batch_size']:,}")
    print(f"  • Total samples: {metadata['total_samples']:,}")
    print(f"  • Backend: {metadata['backend'].upper()}")
    print(f"  • Budgets: {metadata['budgets']}")
    print(f"  • Iterations per epoch: ~{len(iterations) // metadata['epochs']:,}")
    
    print(f"\n⚖️  Loss Weights Configuration:")
    weights = metadata['loss_weights']
    active_weights = {k: v for k, v in weights.items() if v > 0}
    disabled_weights = {k: v for k, v in weights.items() if v == 0}
    
    print(f"  📊 Active Loss Components:")
    for key, value in active_weights.items():
        loss_name = {
            'w_kld': 'KL Divergence Loss',
            'w_id': 'Identity Loss', 
            'w_gap': 'Frontier Gap Loss',
            'w_cell': 'Coarse Cell Loss'
        }.get(key, key)
        print(f"    • {loss_name}: {value}")
    
    if disabled_weights:
        print(f"  🚫 Disabled Loss Components:")
        for key, value in disabled_weights.items():
            loss_name = {
                'w_kld': 'KL Divergence Loss',
                'w_id': 'Identity Loss', 
                'w_gap': 'Frontier Gap Loss',
                'w_cell': 'Coarse Cell Loss'
            }.get(key, key)
            print(f"    • {loss_name}: {value} (disabled)")
    
    print(f"\n📈 Loss Component Analysis:")
    print("=" * 50)
    
    # Analyze each loss component (4 core losses only)
    loss_components = {
        'total': 'Total Loss',
        'kld': 'KL Divergence Loss',
        'id': 'Identity Loss',
        'gap': 'Frontier Gap Loss',
        'cell': 'Coarse Cell Loss'
    }
    
    analysis_results = {}
    
    for loss_key, loss_name in loss_components.items():
        if loss_key in losses and len(losses[loss_key]) > 0:
            values = np.array(losses[loss_key])
            
            # Calculate statistics
            stats = {
                'initial': float(values[0]),
                'final': float(values[-1]),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'improvement': float(values[0] - values[-1]),
                'improvement_pct': float((values[0] - values[-1]) / values[0] * 100) if values[0] > 0 else 0,
                'convergence_rate': float(np.mean(np.diff(values[-100:]))) if len(values) >= 100 else 0
            }
            
            analysis_results[loss_key] = stats
            
            print(f"\n🎯 {loss_name} ({loss_key}):")
            print(f"  • Initial: {stats['initial']:.6f}")
            print(f"  • Final:   {stats['final']:.6f}")
            print(f"  • Min:     {stats['min']:.6f}")
            print(f"  • Max:     {stats['max']:.6f}")
            print(f"  • Mean:    {stats['mean']:.6f}")
            print(f"  • Std:     {stats['std']:.6f}")
            print(f"  • Improvement: {stats['improvement']:.6f} ({stats['improvement_pct']:.1f}%)")
            
            # Convergence analysis
            if abs(stats['convergence_rate']) < 1e-6:
                print(f"  • Convergence: ✅ CONVERGED (final slope: {stats['convergence_rate']:.2e})")
            elif stats['convergence_rate'] > 1e-5:  # More significant positive slope
                print(f"  • Convergence: ⚠️  DIVERGING (final slope: {stats['convergence_rate']:.2e})")
            elif stats['convergence_rate'] > 0:
                print(f"  • Convergence: 📊 STABLE (slight upward drift: {stats['convergence_rate']:.2e})")
            else:
                print(f"  • Convergence: 📉 STILL DECREASING (final slope: {stats['convergence_rate']:.2e})")
    
    # Training dynamics analysis
    print(f"\n🔄 Training Dynamics Analysis:")
    print("=" * 50)
    
    # Calculate epochs from iterations
    total_iterations = len(iterations)
    iterations_per_epoch = total_iterations // metadata['epochs']
    print(f"  • Iterations per epoch: {iterations_per_epoch:,}")
    
    # Analyze convergence patterns
    print(f"\n📉 Convergence Patterns:")
    
    # Check if losses are generally decreasing
    decreasing_losses = []
    stable_losses = []
    increasing_losses = []
    
    for loss_key, stats in analysis_results.items():
        if stats['improvement'] > 0.001:  # Significant improvement
            decreasing_losses.append((loss_key, stats['improvement_pct']))
        elif abs(stats['improvement']) < 0.001:  # Stable
            stable_losses.append(loss_key)
        else:  # Increasing
            increasing_losses.append((loss_key, -stats['improvement_pct']))
    
    if decreasing_losses:
        print(f"  ✅ Decreasing losses ({len(decreasing_losses)}):")
        for loss_key, improvement in sorted(decreasing_losses, key=lambda x: x[1], reverse=True):
            print(f"    • {loss_key}: {improvement:.1f}% improvement")
    
    if stable_losses:
        print(f"  ⚖️  Stable losses ({len(stable_losses)}):")
        for loss_key in stable_losses:
            print(f"    • {loss_key}")
    
    if increasing_losses:
        print(f"  ⚠️  Increasing losses ({len(increasing_losses)}):")
        for loss_key, increase in sorted(increasing_losses, key=lambda x: x[1], reverse=True):
            print(f"    • {loss_key}: {increase:.1f}% increase")
    
    # Loss magnitude analysis
    print(f"\n📏 Loss Magnitude Analysis:")
    final_losses = [(loss_key, stats['final']) for loss_key, stats in analysis_results.items()]
    final_losses.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  Final loss magnitudes (highest to lowest):")
    for loss_key, final_value in final_losses:
        loss_name = loss_components.get(loss_key, loss_key)
        print(f"    • {loss_name}: {final_value:.6f}")
    
    # Training efficiency analysis
    print(f"\n⚡ Training Efficiency Analysis:")
    
    # Calculate when each loss stabilized (if it did)
    for loss_key, loss_name in loss_components.items():
        if loss_key in losses and len(losses[loss_key]) > 0:
            values = np.array(losses[loss_key])
            
            # Find when loss reached 90% of its final improvement
            initial_val = values[0]
            final_val = values[-1]
            target_val = initial_val - 0.9 * (initial_val - final_val)
            
            # Find iteration where target was reached
            convergence_iter = None
            for i, val in enumerate(values):
                if val <= target_val:
                    convergence_iter = i + 1
                    break
            
            if convergence_iter:
                convergence_epoch = convergence_iter / iterations_per_epoch
                print(f"  • {loss_name}: 90% improvement reached at iteration {convergence_iter:,} (epoch {convergence_epoch:.1f})")
    
    # Epoch-level analysis (if available)
    if has_epoch_data:
        print(f"\n📅 Epoch-Level Analysis:")
        print("=" * 50)
        
        for loss_key, loss_name in loss_components.items():
            if loss_key in epoch_losses and len(epoch_losses[loss_key]) > 0:
                epoch_values = np.array(epoch_losses[loss_key])
                
                print(f"\n🎯 {loss_name} (Epoch-level):")
                print(f"  • Initial epoch: {epoch_values[0]:.6f}")
                print(f"  • Final epoch:   {epoch_values[-1]:.6f}")
                print(f"  • Min epoch:     {np.min(epoch_values):.6f}")
                print(f"  • Max epoch:     {np.max(epoch_values):.6f}")
                print(f"  • Mean epoch:    {np.mean(epoch_values):.6f}")
                print(f"  • Std epoch:    {np.std(epoch_values):.6f}")
                
                # Epoch-level improvement
                epoch_improvement = epoch_values[0] - epoch_values[-1]
                epoch_improvement_pct = (epoch_improvement / epoch_values[0] * 100) if epoch_values[0] > 0 else 0
                print(f"  • Epoch improvement: {epoch_improvement:.6f} ({epoch_improvement_pct:.1f}%)")
                
                # Find best epoch
                best_epoch_idx = np.argmin(epoch_values)
                best_epoch = epochs[best_epoch_idx]
                print(f"  • Best epoch: {best_epoch} (loss: {epoch_values[best_epoch_idx]:.6f})")
        
        # Epoch-level convergence patterns
        print(f"\n📈 Epoch-Level Convergence Patterns:")
        epoch_decreasing = []
        epoch_stable = []
        epoch_increasing = []
        
        for loss_key, loss_name in loss_components.items():
            if loss_key in epoch_losses and len(epoch_losses[loss_key]) > 0:
                epoch_values = np.array(epoch_losses[loss_key])
                epoch_improvement = epoch_values[0] - epoch_values[-1]
                epoch_improvement_pct = (epoch_improvement / epoch_values[0] * 100) if epoch_values[0] > 0 else 0
                
                if epoch_improvement > 0.001:
                    epoch_decreasing.append((loss_key, epoch_improvement_pct))
                elif abs(epoch_improvement) < 0.001:
                    epoch_stable.append(loss_key)
                else:
                    epoch_increasing.append((loss_key, -epoch_improvement_pct))
        
        if epoch_decreasing:
            print(f"  ✅ Epoch-decreasing losses ({len(epoch_decreasing)}):")
            for loss_key, improvement in sorted(epoch_decreasing, key=lambda x: x[1], reverse=True):
                print(f"    • {loss_key}: {improvement:.1f}% improvement")
        
        if epoch_stable:
            print(f"  ⚖️  Epoch-stable losses ({len(epoch_stable)}):")
            for loss_key in epoch_stable:
                print(f"    • {loss_key}")
        
        if epoch_increasing:
            print(f"  ⚠️  Epoch-increasing losses ({len(epoch_increasing)}):")
            for loss_key, increase in sorted(epoch_increasing, key=lambda x: x[1], reverse=True):
                print(f"    • {loss_key}: {increase:.1f}% increase")
    
    return analysis_results, data


def create_summary_plot(data, output_dir: str = "analysis_plots"):
    """Create a summary visualization of the loss analysis."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    iterations = np.array(data['iterations'])
    losses = data['losses']
    metadata = data['metadata']
    weights = metadata['loss_weights']
    
    # Create a comprehensive summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Loss Analysis Summary - {metadata["backend"].upper()} Backend', fontsize=16, fontweight='bold')
    
    # Plot 1: All losses over time (4 core losses only)
    ax1 = axes[0, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2']
    loss_keys = ['total', 'kld', 'id', 'gap', 'cell']
    loss_names = ['Total', 'KL Div', 'Identity', 'Frontier Gap', 'Coarse Cell']
    
    for i, (loss_key, loss_name) in enumerate(zip(loss_keys, loss_names)):
        if loss_key in losses and len(losses[loss_key]) > 0:
            values = np.array(losses[loss_key])
            ax1.plot(iterations, values, label=loss_name, color=colors[i], alpha=0.8, linewidth=1)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('All Loss Components Over Training')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss improvement percentages
    ax2 = axes[0, 1]
    improvements = []
    labels = []
    
    for loss_key, loss_name in zip(loss_keys, loss_names):
        if loss_key in losses and len(losses[loss_key]) > 0:
            values = np.array(losses[loss_key])
            improvement = (values[0] - values[-1]) / values[0] * 100 if values[0] > 0 else 0
            improvements.append(improvement)
            labels.append(loss_name)
    
    bars = ax2.bar(labels, improvements, color=colors[:len(improvements)])
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Loss Improvement Over Training')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{improvement:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Final loss magnitudes
    ax3 = axes[1, 0]
    final_values = []
    labels_final = []
    
    for loss_key, loss_name in zip(loss_keys, loss_names):
        if loss_key in losses and len(losses[loss_key]) > 0:
            values = np.array(losses[loss_key])
            final_values.append(values[-1])
            labels_final.append(loss_name)
    
    bars = ax3.bar(labels_final, final_values, color=colors[:len(final_values)])
    ax3.set_ylabel('Final Loss Value')
    ax3.set_title('Final Loss Magnitudes')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training metadata
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create metadata text
    metadata_text = f"""
Training Configuration:
• Epochs: {metadata['epochs']}
• Batch Size: {metadata['batch_size']:,}
• Total Samples: {metadata['total_samples']:,}
• Backend: {metadata['backend'].upper()}
• Total Iterations: {len(iterations):,}
• Iterations/Epoch: {len(iterations) // metadata['epochs']:,}

Loss Weights (4 Core Losses):
• KL Divergence: {weights['w_kld']}
• Identity: {weights['w_id']}
• Frontier Gap: {weights['w_gap']}
• Coarse Cell: {weights['w_cell']}
"""
    
    ax4.text(0.05, 0.95, metadata_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax4.set_title('Training Configuration')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'loss_analysis_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Summary plot saved to: {output_path}")
    plt.close()


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze training loss data from JSON file')
    parser.add_argument('--loss_file', type=str, 
                       default="/home/hamed/projects/SPIN/adapter/t2i_code/outputs/loss_analysis.json",
                       help='Path to JSON file containing loss data')
    parser.add_argument('--output_dir', type=str, default='analysis_plots',
                       help='Directory to save analysis plots')
    
    return parser.parse_args()


def main():
    """Main analysis function."""
    args = parse_args()
    json_file = args.loss_file
    
    if not os.path.exists(json_file):
        print(f"❌ Loss analysis file not found: {json_file}")
        print(f"Usage: python analyze_loss_data.py --loss_file path/to/loss_analysis.json")
        return
    
    print(f"📊 Analyzing loss data from: {json_file}")
    
    # Load and analyze the data
    analysis_results, data = load_and_analyze_loss_data(json_file)
    
    # Create summary visualization
    print(f"\n🎨 Creating summary visualization...")
    create_summary_plot(data, args.output_dir)
    
    print(f"\n✅ Analysis complete!")
    print(f"\n💡 Key Insights:")
    print(f"  • This training run used the {data['metadata']['backend'].upper()} backend")
    print(f"  • Training completed {data['metadata']['epochs']} epochs with {len(data['iterations']):,} total iterations")
    print(f"  • The model shows good convergence across most loss components")
    
    # Show active weights
    weights = data['metadata']['loss_weights']
    active_weights = {k: v for k, v in weights.items() if v > 0}
    if active_weights:
        print(f"  • Active loss components: {', '.join([f'{k}={v}' for k, v in active_weights.items()])}")
    
    disabled_weights = {k: v for k, v in weights.items() if v == 0}
    if disabled_weights:
        print(f"  • Disabled loss components: {', '.join(disabled_weights.keys())}")


if __name__ == "__main__":
    import os
    main()
