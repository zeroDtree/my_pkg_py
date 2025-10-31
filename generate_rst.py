#!/usr/bin/env python3
"""
Script to automatically generate rst documentation files for all ls_mlkit modules
"""

import os
import re
import argparse
from pathlib import Path

def get_module_title(filename, module_type='utils'):
    """Generate title based on filename and module type"""
    # Remove .py extension
    name = filename.replace('.py', '')
    
    # Handle special naming based on module type
    title_maps = {
        'utils': {
            'cgraph': 'CGraph Utils',
            'cuda': 'CUDA Utils', 
            'decorators': 'Decorators',
            'dequantize': 'Dequantize Utils',
            'hash': 'Hash Utils',
            'huggingface': 'HuggingFace Utils',
            'image': 'Image Utils',
            'interp': 'Interpolation Utils',
            'iterator': 'Iterator Utils',
            'llm': 'LLM Utils',
            'log': 'Logging Utils',
            'lora': 'LoRA Utils',
            'observer': 'Observer Utils',
            'proxy': 'Proxy Utils',
            'resource_monitor': 'Resource Monitor',
            'scheduler': 'Scheduler Utils',
            'se3': 'SE(3) Utils',
            'seed': 'Seed Utils',
            'shape': 'Shape Utils',
            'show': 'Show Utils',
            'sniffer': 'Sniffer Utils',
            'vector_utils': 'Vector Utils',
            # Submodules
            'lie_group': 'Lie Group Utils',
            'riemannian_manifold': 'Riemannian Manifold Utils',
            'so3_utils': 'SO(3) Utils',
            'so3': 'SO(3) Utils',
            'bio_masker': 'Bio Masker',
            'image_masker': 'Image Masker',
            'masker_interface': 'Masker Interface',
            'anm': 'ANM Utils',
            'force_fields': 'Force Fields',
            'nma': 'NMA Utils',
            'context': 'Context Utils',
            'forward_backward_offload': 'Forward Backward Offload',
            'forward_hook': 'Forward Hook',
            'gradient_offload_v2': 'Gradient Offload V2',
            'gradient_offload': 'Gradient Offload',
            'graph_hook': 'Graph Hook',
            'model_offload': 'Model Offload',
            'saved_tensor_offload': 'Saved Tensor Offload',
            'split': 'Split Utils'
        },
        'diffuser': {
            'base_diffuser': 'Base Diffuser',
            'euclidean_diffuser': 'Euclidean Diffuser',
            'euclidean_ddpm_diffuser': 'Euclidean DDPM Diffuser',
            'euclidean_ddim_diffuser': 'Euclidean DDIM Diffuser',
            'manifold_diffuser': 'Manifold Diffuser',
            'lie_group_diffuser': 'Lie Group Diffuser',
            'so3_diffuser': 'SO(3) Diffuser',
            'loss_utils': 'Loss Utils',
            'model_interface': 'Model Interface',
            'time_scheduler': 'Time Scheduler',
            # SDE submodules
            'base_sde': 'Base SDE',
            'corrector': 'Corrector',
            'predictor': 'Predictor',
            'sampler': 'Sampler',
            'score_fn_utils': 'Score Function Utils',
            'sde_lib': 'SDE Library',
            # Conditioner submodules
            'conditioner': 'Conditioner',
            'nma_conditioner': 'NMA Conditioner'
        },
        'dataset': {
            'iris': 'Iris Dataset',
            'lda_dataset': 'LDA Dataset',
            'minist_cifar': 'MNIST CIFAR Dataset',
            'MT19937': 'MT19937 Random Generator',
            'nature_language': 'Natural Language Dataset',
            'regular_language': 'Regular Language Dataset'
        },
        'model': {
            'longLinear': 'Long Linear Model',
            'causal_transformer': 'Causal Transformer',
            'generate': 'Generation Utils',
            'tokenizer': 'Tokenizer'
        },
        'optimizer': {
            'sam': 'SAM Optimizer',
            'kfa_optimizer': 'KFA Optimizer',
            'kfa': 'KFA Implementation'
        },
        'pipeline': {
            'pipeline': 'Base Pipeline',
            'distributed_pipeline': 'Distributed Pipeline',
            'dist_pipeline_impl': 'Distributed Pipeline Implementation'
        },
        'scheduler': {
            'lr_scheduler_factory': 'Learning Rate Scheduler Factory'
        }
    }
    
    title_map = title_maps.get(module_type, {})
    return title_map.get(name, name.replace('_', ' ').title())

def generate_rst_content(module_path, module_name, package_name, module_type='utils'):
    """Generate rst file content"""
    title = get_module_title(module_name, module_type)
    
    # Generate module path
    if '/' in module_path:
        # Submodule
        parts = module_path.split('/')
        # Remove .py extension from each part
        parts = [part.replace('.py', '') for part in parts]
        full_module_name = f"ls_mlkit.{package_name}.{'.'.join(parts)}"
    else:
        # Root module
        full_module_name = f"ls_mlkit.{package_name}.{module_name}"
    
    content = f"""{title}
{'=' * len(title)}

.. automodule:: {full_module_name}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    return content

def generate_index_rst_content(package_name, module_files, subdir_files=None):
    """Generate index.rst file content for a package"""
    # Package title mapping
    package_titles = {
        'my_utils': 'Utility Module',
        'my_diffuser': 'Diffuser Module', 
        'my_dataset': 'Dataset Module',
        'my_model': 'Model Module',
        'my_optimizer': 'Optimizer Module',
        'my_pipeline': 'Pipeline Module',
        'my_scheduler': 'Scheduler Module'
    }
    
    title = package_titles.get(package_name, package_name.replace('my_', '').replace('_', ' ').title() + ' Module')
    
    # Package descriptions
    descriptions = {
        'my_utils': 'The utility module provides various helper functions and utilities.',
        'my_diffuser': 'The diffuser module contains implementations of various diffusion models.',
        'my_dataset': 'The dataset module provides various datasets and data utilities.',
        'my_model': 'The model module contains neural network model implementations.',
        'my_optimizer': 'The optimizer module provides custom optimization algorithms.',
        'my_pipeline': 'The pipeline module contains training and inference pipeline implementations.',
        'my_scheduler': 'The scheduler module provides learning rate scheduling utilities.'
    }
    
    description = descriptions.get(package_name, f'The {package_name.replace("my_", "")} module.')
    
    content = f"""{title}
{'=' * len(title)}

{description}

.. toctree::
   :maxdepth: 1

"""
    
    # Add module files
    for module_file in sorted(module_files):
        rst_name = module_file.replace('.py', '')
        content += f"   {rst_name}\n"
    
    # Add subdirectory index files if any
    if subdir_files:
        for subdir in sorted(subdir_files.keys()):
            content += f"   {subdir}/index\n"
    
    return content

def generate_main_api_index_content(available_packages):
    """Generate main API index.rst file content"""
    content = """API Reference
=============

This section contains the complete API reference for ls-mlkit.

.. toctree::
   :maxdepth: 1

"""
    
    # Package order mapping for consistent ordering
    package_order = {
        'my_dataset': 'dataset/index',
        'my_diffuser': 'diffuser/index', 
        'my_model': 'model/index',
        'my_optimizer': 'optimizer/index',
        'my_pipeline': 'pipeline/index',
        'my_scheduler': 'scheduler/index',
        'my_utils': 'utils/index'
    }
    
    # Add packages in the specified order
    for package_name in ['my_dataset', 'my_diffuser', 'my_model', 'my_optimizer', 'my_pipeline', 'my_scheduler', 'my_utils']:
        if package_name in available_packages:
            module_name = package_name.replace('my_', '')
            content += f"   {module_name}/index\n"
    
    return content

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Automatically generate rst documentation files for all ls_mlkit modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python generate_rst.py                                    # Generate all files for all packages (with index files)
  python generate_rst.py --package my_utils                # Generate only my_utils package (with index files)
  python generate_rst.py --package my_diffuser             # Generate only my_diffuser package (with index files)
  python generate_rst.py --module cuda --package my_utils  # Generate only cuda module in my_utils
  python generate_rst.py --subdir manifold --package my_utils # Generate only manifold subdirectory
  python generate_rst.py --dry-run                         # Preview mode, don't actually generate files
  python generate_rst.py --no-generate-index               # Skip generating package index.rst files
  python generate_rst.py --no-generate-main-index          # Skip generating main API index.rst file
        """
    )
    
    parser.add_argument(
        '--src-dir', 
        type=str, 
        default="/home/zengls/repo/my_pkg_py/src/ls_mlkit",
        help="Source code directory path (default: /home/zengls/repo/my_pkg_py/src/ls_mlkit)"
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default="/home/zengls/repo/my_pkg_py/docs/source/api",
        help="RST files output directory (default: /home/zengls/repo/my_pkg_py/docs/source/api)"
    )
    
    parser.add_argument(
        '--package', 
        type=str, 
        help="Generate rst files for specified package only (e.g., my_utils, my_diffuser, my_dataset)"
    )
    
    parser.add_argument(
        '--module', 
        type=str, 
        help="Generate rst file for specified module only (e.g., cuda, cgraph)"
    )
    
    parser.add_argument(
        '--subdir', 
        type=str, 
        help="Generate rst files for specified subdirectory only (e.g., manifold, mask, nma, offload, sde, conditioner)"
    )
    
    parser.add_argument(
        '--generate-index', 
        action='store_true',
        default=True,
        help="Generate index.rst files for each package (default: True)"
    )
    
    parser.add_argument(
        '--no-generate-index', 
        dest='generate_index',
        action='store_false',
        help="Skip generating index.rst files for each package"
    )
    
    parser.add_argument(
        '--generate-main-index', 
        action='store_true',
        default=True,
        help="Generate main API index.rst file (default: True)"
    )
    
    parser.add_argument(
        '--no-generate-main-index', 
        dest='generate_main_index',
        action='store_false',
        help="Skip generating main API index.rst file"
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help="Preview mode, show files to be generated without actually creating them"
    )
    
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help="Force overwrite existing rst files"
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help="Show verbose information"
    )
    
    return parser.parse_args()

def collect_python_files(src_dir, package_name, module_filter=None, subdir_filter=None):
    """Collect Python files for a specific package"""
    python_files = []
    package_dir = src_dir / package_name
    
    if not package_dir.exists():
        return python_files
    
    # Python files in root directory
    for file in package_dir.glob("*.py"):
        if file.name != "__init__.py":
            if module_filter is None or file.stem == module_filter:
                python_files.append((file.name, file.name))
    
    # Python files in subdirectories
    for subdir in package_dir.iterdir():
        if subdir.is_dir() and subdir.name != "__pycache__":
            if subdir_filter is None or subdir.name == subdir_filter:
                for file in subdir.glob("*.py"):
                    if file.name != "__init__.py":
                        relative_path = f"{subdir.name}/{file.name}"
                        python_files.append((relative_path, file.name))
    
    return python_files

def collect_all_packages(src_dir, package_filter=None):
    """Collect all available packages"""
    packages = []
    
    for item in src_dir.iterdir():
        if item.is_dir() and item.name.startswith('my_') and item.name != '__pycache__':
            if package_filter is None or item.name == package_filter:
                packages.append(item.name)
    
    return sorted(packages)

def collect_subdirectories(src_dir, package_name):
    """Collect subdirectories for a package"""
    subdirs = {}
    package_dir = src_dir / package_name
    
    if not package_dir.exists():
        return subdirs
    
    for subdir in package_dir.iterdir():
        if subdir.is_dir() and subdir.name != "__pycache__":
            subdir_files = []
            for file in subdir.glob("*.py"):
                if file.name != "__init__.py":
                    subdir_files.append(file.name)
            if subdir_files:
                subdirs[subdir.name] = subdir_files
    
    return subdirs

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set paths
    src_dir = Path(args.src_dir)
    rst_output_dir = Path(args.output_dir)
    
    if not src_dir.exists():
        print(f"Error: Source directory does not exist: {src_dir}")
        return 1
    
    if args.verbose:
        print(f"Source directory: {src_dir}")
        print(f"Output directory: {rst_output_dir}")
    
    # Ensure output directory exists
    rst_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect packages to process
    packages = collect_all_packages(src_dir, args.package)
    
    if not packages:
        print("No packages found matching the criteria")
        return 1
    
    print(f"Found packages: {', '.join(packages)}")
    
    total_generated = 0
    total_skipped = 0
    
    for package_name in packages:
        print(f"\n=== Processing package: {package_name} ===")
        
        # Determine module type for title generation
        module_type = package_name.replace('my_', '')
        
        # Set package output directory
        package_output_dir = rst_output_dir / module_type
        package_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect Python files for this package
        python_files = collect_python_files(
            src_dir, 
            package_name,
            args.module, 
            args.subdir
        )
        
        if not python_files:
            print(f"No Python files found in {package_name}")
            continue
        
        print(f"Found {len(python_files)} module files in {package_name}")
        
        if args.dry_run:
            print(f"Preview mode - Files to be generated for {package_name}:")
            for module_path, module_name in python_files:
                rst_filename = module_name.replace('.py', '.rst')
                print(f"  - {module_type}/{rst_filename}")
            continue
        
        # Generate rst files for this package
        generated_files = []
        skipped_files = []
        
        # Collect subdirectories for potential index generation
        subdirs = collect_subdirectories(src_dir, package_name)
        
        # Generate subdirectory index files if needed
        if subdirs and args.generate_index:
            for subdir_name, subdir_files in subdirs.items():
                subdir_output_dir = package_output_dir / subdir_name
                subdir_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate index.rst for subdirectory
                index_content = generate_index_rst_content(f"{package_name}_{subdir_name}", subdir_files)
                index_filepath = subdir_output_dir / "index.rst"
                
                if not index_filepath.exists() or args.force:
                    with open(index_filepath, 'w', encoding='utf-8') as f:
                        f.write(index_content)
                    print(f"Generated index: {module_type}/{subdir_name}/index.rst")
        
        # Generate individual module files
        root_files = []
        for module_path, module_name in python_files:
            # Generate rst filename
            rst_filename = module_name.replace('.py', '.rst')
            
            # Determine output path based on whether it's in a subdirectory
            if '/' in module_path:
                subdir_name = module_path.split('/')[0]
                subdir_output_dir = package_output_dir / subdir_name
                subdir_output_dir.mkdir(parents=True, exist_ok=True)
                rst_filepath = subdir_output_dir / rst_filename
            else:
                rst_filepath = package_output_dir / rst_filename
                root_files.append(module_name)
            
            # Check if file already exists
            if rst_filepath.exists() and not args.force:
                skipped_files.append(rst_filename)
                if args.verbose:
                    print(f"Skipped (already exists): {rst_filename}")
                continue
            
            # Generate content
            content = generate_rst_content(
                module_path, 
                module_name.replace('.py', ''), 
                package_name,
                module_type
            )
            
            # Write file
            with open(rst_filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            generated_files.append(rst_filename)
            print(f"Generated: {module_type}/{rst_filename}")
        
        # Generate package index.rst if requested
        if args.generate_index:
            index_content = generate_index_rst_content(package_name, root_files, subdirs)
            index_filepath = package_output_dir / "index.rst"
            
            if not index_filepath.exists() or args.force:
                with open(index_filepath, 'w', encoding='utf-8') as f:
                    f.write(index_content)
                print(f"Generated index: {module_type}/index.rst")
        
        if generated_files:
            print(f"Successfully generated {len(generated_files)} rst files for {package_name}")
            total_generated += len(generated_files)
        
        if skipped_files:
            print(f"Skipped {len(skipped_files)} existing files in {package_name}")
            total_skipped += len(skipped_files)
    
    if args.dry_run:
        return 0
    
    # Generate main API index.rst if requested
    if args.generate_main_index:
        print(f"\n=== Generating main API index ===")
        main_index_content = generate_main_api_index_content(packages)
        main_index_filepath = rst_output_dir / "index.rst"
        
        if not main_index_filepath.exists() or args.force:
            with open(main_index_filepath, 'w', encoding='utf-8') as f:
                f.write(main_index_content)
            print(f"Generated main API index: api/index.rst")
        else:
            print(f"Skipped main API index (already exists): api/index.rst")
            print("Use --force parameter to force overwrite")
    
    print(f"\n=== Summary ===")
    print(f"Total generated: {total_generated} files")
    if total_skipped > 0:
        print(f"Total skipped: {total_skipped} files")
        print("Use --force parameter to force overwrite existing files")
    
    print(f"\nAll files generated to: {rst_output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
