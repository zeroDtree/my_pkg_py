"""
Comparison plotting utilities for multi-model analysis
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_entity_wise_comparison(
    results_dict: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
    figsize: Tuple[float, float] = (12, 8),
    top_n: int = 20,
    name_formatter: Optional[Callable[[str], str]] = None,
    entity_key: str = "entity_name",
    value_key: str = "value",
    title: Optional[str] = None,
    xlabel: str = "Entity Names",
    ylabel: str = "Value",
    filename: str = "entity_wise_comparison.png",
    name_truncate_length: int = 15,
    alpha: float = 0.7,
    dpi: int = 300,
    grid: bool = True,
    grid_alpha: float = 0.3,
    legend_bbox_to_anchor: Tuple[float, float] = (0.5, 1.2),
    legend_loc: str = "upper center",
    legend_fontsize: Union[int, str] = 7,
    legend_ncol: int = 1,
    rotation: float = 45,
    ha: str = "right",
    show: bool = False,
    save: bool = True,
    return_fig_ax: bool = False,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:

    # Extract all unique entity names
    all_entity_names = set()
    for model_data in results_dict.values():
        for entry in model_data:
            if entity_key in entry:
                all_entity_names.add(entry[entity_key])

    all_entities = sorted(all_entity_names)

    # Calculate variation for each entity across models
    entity_variations = {}
    for entity in all_entities:
        values = []
        for model_data in results_dict.values():
            # Find value for this entity in this model
            for entry in model_data:
                if entry.get(entity_key) == entity and value_key in entry:
                    values.append(entry[value_key])
                    break

        # Only consider entities present in multiple models
        if len(values) > 1:
            entity_variations[entity] = np.std(values)

    # Select top N entities by variation
    sorted_entities = sorted(entity_variations.items(), key=lambda x: x[1], reverse=True)
    selected_entities = [entity for entity, _ in sorted_entities[:top_n]]

    if not selected_entities:
        print("No entities found with variation across models.")
        return None

    # Create figure with dynamic width based on number of entities
    fig_width = max(figsize[0], len(selected_entities) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))

    x_positions = np.arange(len(selected_entities))
    bar_width = 0.8 / len(results_dict)

    # Generate colors for different models
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_dict)))

    for i, ((model_name, data), color) in enumerate(zip(results_dict.items(), colors)):
        model_values = []
        for entity in selected_entities:
            # Find value for this entity in this model
            value = None
            for entry in data:
                if entry.get(entity_key) == entity and value_key in entry:
                    value = entry[value_key]
                    break
            model_values.append(value if value is not None else 0.0)

        # Plot bars for this model
        label = name_formatter(model_name) if name_formatter else model_name
        x_pos = x_positions + i * bar_width - (len(results_dict) - 1) * bar_width / 2
        ax.bar(x_pos, model_values, bar_width, label=label, color=color, alpha=alpha)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"Entity-wise {ylabel} Comparison (Top {top_n} by Variation)"
    ax.set_title(title)

    ax.set_xticks(x_positions)

    # Truncate entity names for readability
    truncated_names = [
        name[:name_truncate_length] + "..." if len(name) > name_truncate_length else name for name in selected_entities
    ]
    ax.set_xticklabels(truncated_names, rotation=rotation, ha=ha)

    # Add legend and grid
    ax.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc, fontsize=legend_fontsize, ncol=legend_ncol)

    if grid:
        ax.grid(True, alpha=grid_alpha, axis="y")

    plt.tight_layout()

    # Save the plot
    if save:
        output_path = os.path.join(output_dir, filename)
        save_kwargs = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.2}

        # Include legend in saved area
        legend = ax.get_legend()
        if legend:
            save_kwargs["bbox_extra_artists"] = [legend]

        plt.savefig(output_path, **save_kwargs)
        print(f"Entity-wise comparison plot saved to: {output_path}")

    # Show the plot
    if show:
        plt.show()

    # Return figure and axes if requested
    if return_fig_ax:
        return fig, ax

    # Close the figure if not returning it
    if not return_fig_ax:
        plt.close(fig)


if __name__ == "__main__":
    # Example usage and testing
    import numpy as np

    # Generate example data
    np.random.seed(42)

    example_results = {
        "Model_A": [{"entity_name": f"item_{i}", "value": np.random.normal(80, 10)} for i in range(50)],
        "Model_B": [{"entity_name": f"item_{i}", "value": np.random.normal(75, 12)} for i in range(50)],
        "Model_C": [{"entity_name": f"item_{i}", "value": np.random.normal(85, 8)} for i in range(50)],
    }

    # Test the function
    plot_entity_wise_comparison(example_results, output_dir=".", top_n=10, show=True, filename="test_comparison.png")
