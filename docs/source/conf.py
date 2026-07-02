import jubik

needs_sphinx = '3.2.0'

extensions = [
    'sphinx.ext.napoleon',    # Support for NumPy and Google style docstrings
    'sphinx.ext.mathjax',     # Render math as images
    'sphinx.ext.viewcode',    # Add links to highlighted source code
    'sphinx.ext.intersphinx', # Links to other sphinx docs (mostly numpy)
    "sphinx.ext.autodoc",
    'myst_parser',            # Parse markdown
    'sphinxcontrib.bibtex',
] 

bibtex_bibfiles = ['user/paper.bib']
master_doc = 'index'

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "strikethrough",
    "tasklist",
]

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

intersphinx_mapping = {"numpy": ("https://numpy.org/doc/stable/", None),
                       "ducc0": ("https://mtr.pages.mpcdf.de/ducc/", None),
                       "scipy": ('https://docs.scipy.org/doc/scipy/reference/', None),
                       }

autodoc_default_options = {'special-members': '__init__'}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True
napoleon_include_special_with_doc = True

project = u'jubik'
copyright = u'2020-2025, Max-Planck-Society'
author = u'Vincent Eberle, Matteo Guardiani, Margret Westerkamp'

release = jubik.version.__version__
version = release[:-2]

language = "en"
exclude_patterns = []
add_module_names = False

html_theme = "pydata_sphinx_theme"
html_logo = 'ubik-logo.jpg'

html_theme_options = {
    "logo": {
        "image_light": "ubik-logo.jpg",
        "image_dark": "ubik-logo.jpg"
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NIFTy-PPL/J-UBIK",
            "icon": "fab fa-github",
        }
    ],
    "navbar_persistent": ["search-field", "theme-switcher"],
    "navbar_end": ["navbar-icon-links"],
}

html_last_updated_fmt = '%b %d, %Y'

exclude_patterns = ['mod/modules.rst']
