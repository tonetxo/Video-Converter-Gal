import logging
import argparse
import os
from .utils import verify_tools
from .interface import create_interface

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--appdir', type=str, default=None, help='Path to the AppDir, for internal use in AppImage.')
    args = parser.parse_args()

    # If running inside an AppImage, change the current working directory to the AppDir's usr directory
    # This ensures that relative paths within the code work correctly.
    if args.appdir:
        os.chdir(os.path.join(args.appdir, 'usr'))

    verify_tools()
    demo = create_interface()
    # When running from AppImage, we want the browser to open, but let the user decide the server name.
    # The AppRun script handles the LAUNCH_CWD.
    demo.launch(inbrowser=True, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
