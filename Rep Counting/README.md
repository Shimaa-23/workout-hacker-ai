# Rep Counting Module

Contains:

* `Combined_model.pth` trained model
* `rep_counter_interface.py` inference + rep counting logic
* `requirements.txt`

## Usage

```python id="j3n7ud"
from rep_counter_interface import RepCounterInterface
rc = RepCounterInterface(model_path="Combined_model.pth")
```

Returns:

 exercise
 reps
 confidence
 phase
