import subprocess
import sys
from pkg_resources import working_set, parse_version

def get_installed_packages():
    """Get dictionary of installed packages and their versions."""
    return {pkg.key: pkg.version for pkg in working_set}

def parse_requirements(filename):
    """Parse requirements file into package names and versions."""
    requirements = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '==' in line:
                    name, version = line.split('==')
                    requirements[name.lower()] = version
                else:
                    requirements[line.lower()] = None
    return requirements

def main():
    print("Checking for missing packages...")
    
    # Get currently installed packages
    installed = get_installed_packages()
    
    # Read requirements
    required = parse_requirements('requirements.txt')
    
    # Find missing packages
    to_install = []
    already_installed = []
    
    for package, required_version in required.items():
        if package not in installed:
            if required_version:
                to_install.append(f"{package}=={required_version}")
            else:
                to_install.append(package)
        else:
            already_installed.append(f"{package} (version {installed[package]})")
    
    # Print summary
    if already_installed:
        print("\nAlready installed packages:")
        for pkg in already_installed:
            print(f"✓ {pkg}")
    
    if not to_install:
        print("\nAll required packages are already installed!")
        return
    
    print(f"\nMissing packages to install ({len(to_install)}):")
    for package in to_install:
        print(f"• {package}")
    
    # Ask for confirmation
    response = input("\nDo you want to install these packages? (y/n): ")
    if response.lower() != 'y':
        print("Installation cancelled.")
        return
    
    print("\nInstalling missing packages...")
    for package in to_install:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing {package}: {e}")

if __name__ == "__main__":
    main() 