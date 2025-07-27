import sys
import json
import params_parser

def main():  
  # load the JSON file with hyperparameters
  with open("config_params.json") as f:
      data = json.load(f)
  try:
      config = params_parser.HyperparamValidator(**data)
      print("Config validated successfully!")
  except Exception as e:
      print(f"Validation error: {e}")

if __name__ == "__main__":
    sys.exit(main())