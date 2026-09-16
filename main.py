"""SciLifeLab Serve Gradio entry point.

Serve's Gradio launcher executes /app/main.py directly. The real application
implementation lives in the installed histoseg package.
"""

from histoseg.spatial_pathologist.serve_app import main


if __name__ == "__main__":
    main()
