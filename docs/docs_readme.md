### README for Sphinx Documentation

#### Introduction
This README will guide you through setting up and using Sphinx to document your project. The goal is to generate HTML documentation from the provided configuration file `conf.py`.

#### Prerequisites
1. **Python**: Ensure Python is installed on your system.
2. **Environment**: It is recommended to use a virtual environment to manage dependencies.
   Create and activate a virtual environment:
      ```bash
      python -m venv .venv
      source .venv/bin/activate
      ```
   It should say `(.venv)` in your terminal prompt, indicating that the virtual environment is active.
   
   Install the dependencies listed in `requirements.txt`:
      ```bash
      pip install -r requirements.txt
      ```
3. **Sphinx**: Install Sphinx and any necessary extensions:
   ```bash
   pip install -U sphinx
   ```

#### Step 1: Directory Structure
Ensure your project directory structure looks like this:
```
MD_DL_DPP8-9/
├── docs/
│   ├── _build/
│   ├── _static/
│   ├── _templates/
│   ├── conf.py
│   ├── index.rst
│   └── make.bat
├── src/
│   └── your_project_code_here
└── README.md
```

#### Step 2: Configuration File (`conf.py`)
The `conf.py` file is the configuration file for Sphinx. Ensure it is located in the `docs/source` directory. This file sets up project information, general configuration options, and HTML output settings.

#### Step 3: Editing the `conf.py`
1. **Project Information**: Modify the `project`, `copyright`, `author`, and `release` variables to fit your project.
2. **Extensions**: The `extensions` variable lists Sphinx extensions to use. The provided file includes `sphinx.ext.autodoc` and `sphinx.ext.napoleon` for documentation generation.
3. **Paths**: Ensure `PROJECT_ROOT` points to the root of your project directory and that `sys.path.insert(0, PROJECT_ROOT)` is correctly set.
#### Step 4: css files
To customize the appearance of your documentation, you can add custom CSS files. Place your CSS files in the `docs/source/_static` directory and reference them in the `conf.py` file. `custom.css` is already present here, and you can modify it to change the visual style of your documentation (e.g., colors, fonts, layout, line_wrapping).
#### Step 5: Source Files
Sphinx uses reStructuredText files to generate documentation. Common file is `index.rst` in the `docs/` directory. This file typically includes the table of contents for your documentation.
Change the `index.rst` file to include your project's modules and packages. For example:
```rst
.. toctree::
   :maxdepth: 6
   :caption: Contents:

   src.data.data_loading
   src.data.data_preprocessing
   src.config
   src.models
   src.experiments
   src.interpretability
   src.utils
```
#### Step 6: Generating Source files
To generate the reStructuredText files for your source code, you can use the `sphinx-apidoc` command. This command will automatically create `.rst` files for your modules and packages. Run the following command from the `docs/` directory:
```sh
sphinx-apidoc -o ./source/ ../src
```
Here, `-o ./source/` specifies the output directory for the generated `.rst` files, and `../src` points to the source code directory. After running this command, you should see new/updated `.rst` files in the `docs/source/` directory corresponding to your project's modules and packages.

#### Step 7: Generating Documentation
To build the documentation, after ensuring that all configurations and source files are set up correctly. Make sure you are in the `docs/` directory and run the following command:
1. Run the Sphinx build command:
   ```sh
   make html
   ```
   This command will generate the HTML documentation in the `_build/html/` directory.

#### Step 8: Viewing Documentation
After running the build command, you can view the generated documentation by opening the `index.html` file in the `_build/html/` directory.

### Additional Commands
- **Clean Build**:
  ```sh
  make clean
  ```
  This command removes the `_build/` directory and its contents.

- **View Logs**:
  To see detailed logs during the build process, run:
  ```sh
  make html VERBOSE=1
  ```

### Notes
- Ensure all necessary source code files are in the `src/` directory and importable from `conf.py`.


