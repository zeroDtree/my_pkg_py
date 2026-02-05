#!/usr/bin/env bash

# Root source directory
SRC_DIR="src/ls_mlkit"

# Root output directory for UML diagrams
UML_ROOT="docs/uml"

# Create root UML directory if not exists
mkdir -p "$UML_ROOT"

# Manually specify subdirectories to generate UML
PACKAGES=("pipeline" "diffusion" "flow_matching")

for pkg_name in "${PACKAGES[@]}"; do
	pkg_dir="$SRC_DIR/$pkg_name"

	# Check if directory exists and is a Python package
	if [ -d "$pkg_dir" ] && [ -f "$pkg_dir/__init__.py" ]; then
		out_dir="$UML_ROOT/$pkg_name"

		echo "Generating UML for package: $pkg_name"

		mkdir -p "$out_dir"

		uv run pyreverse \
			--output-directory "$out_dir" \
			--output png \
			--no-standalone \
			"$pkg_dir"
	else
		echo "Skipping $pkg_name: Not a valid Python package"
	fi
done

echo "UML generation finished."
