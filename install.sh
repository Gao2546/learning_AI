#!/bin/bash

if [!command -v "sudo" &> /dev/null]; then
    apt-get sudo || sudo apt-get update || apt-get upgrade
fi

if [!command -v "curl" &> /dev/null]; then
    sudo apt-get install curl
fi

if [!command -v "git" &> /dev/null]; then
    sudo apt-get install git
fi

if [!command -v "python3" &> /dev/null]; then
    sudo apt-get install python3
fi

if [!command -v "pip" &> /dev/null]; then
    sudo apt-get install python3-pip
fi

if [!command -v "ca-certificates" &> /dev/null]; then
    sudo apt-get install ca-certificates
fi

# Initialize variables with default values
# VERBOSE=false
# OUTPUT_FILE=""
# INPUT_FILE=""
# REMAINING_ARGS=() # Array to hold non-option arguments
INPUT=false

# Loop while there are still arguments ($# is greater than 0)
while [[ $# -gt 0 ]]; do
  # Get the first argument
  key="$1"

  # Use a case statement to check the argument
  case $key in
    -i|--install)
    #   INPUT_FILE="$2" # The option's value is the *next* argument
      INPUT=true
      shift # Remove the option (-i or --input)
    #   shift # Remove the value (the filename)
      ;;
    # -o|--output)
    #   OUTPUT_FILE="$2"
    #   shift # Remove -o or --output
    #   shift # Remove the filename
    #   ;;
    # -v|--verbose)
    #   VERBOSE=true
    #   shift # Remove -v or --verbose (it has no value)
    #   ;;
    --) # End of options marker
      shift # Remove the --
      break # Stop processing options, remaining args are positional
      ;;
    -*)
      # Unknown option
      echo "Error: Unknown option '$1'"
      exit 1
      ;;
    *)
      # Not an option, save it as a remaining argument
      if [$INPUT = true]; then
        if [$1 = "all"]; then
            . installenv.sh
            . installdocker.sh
            . installpostgres.sh
            . installnode.sh
            INPUT=false
            break
        else
            install_key = $1
            
            case $install_key in
                "env")
                    . installenv.sh
                    ;;
                "docker")
                    . installdocker.sh
                    ;;
                "postgres")
                    . installpostgres.sh
                    ;;
                "node")
                    . installnode.sh
                    ;;
                *)
                    echo "Error: Unknown option '$1'"
                    exit 1
                    ;;
            esac
        fi
      fi
        # INPUT_FILE="$1"
        # INPUT=false
    #   REMAINING_ARGS+=("$1")
      shift # Remove the processed argument
      ;;
  esac
done