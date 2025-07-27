import sys
import json
from params_parser import HyperparamValidator, HyperparamCombinator

def main():  
  # load the JSON file with hyperparameters
  with open("config_params.json") as f:
      data = json.load(f)
  try:
      validated = HyperparamValidator(**data)
      combhandler = HyperparamCombinator(validated)
      combinations = combhandler.generate_combinations()
      for i in combinations:
          print(i)
  except Exception as e:
      print(f"Validation error: {e}")

if __name__ == "__main__":
    sys.exit(main())