"""Main entry point"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Launch Streamlit application
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import os

    # Set Streamlit app path
    app_path = project_root / "ui" / "app_stream_full.py"

    # Run Streamlit
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())
