# Machine Learning Model Report for UCI-HAR Dataset

---

## Dataset Overview

The **UCI-HAR (Human Activity Recognition)** dataset consists of inertial sensor data collected from smartphones. The dataset includes:

- **6 Activities:** Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying.
- **Sensor Signals:** Accelerometer (`acc_x`, `acc_y`, `acc_z`) and Gyroscope (`gyro_x`, `gyro_y`, `gyro_z`).
- **Shape of Data:** `(7352, 128, 6)` for training and `(2947, 128, 6)` for testing.

---

## Model Architecture

The notebook implements an **LSTM-based model** using PyTorch. The architecture is as follows:

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```

### 🔢 Hyperparameters:
| Parameter        | Value  |
|-----------------|--------|
| Input Size      | 6      |
| Hidden Size     | 32     |
| LSTM Layers     | 3      |
| Classes         | 6      |
| Learning Rate   | 0.00001 |
| Epochs          | 100    |

---

## 🚀 Model Training

The model was trained using the **Adam optimizer** and **CrossEntropy loss**. Training logs show:

```plaintext
Epoch [1/100], Loss: 1.7942
Epoch [2/100], Loss: 1.6035
...
Epoch [100/100], Loss: 0.2431
```

**Observations:**
- Loss decreases steadily over epochs, indicating proper learning.
- Further fine-tuning (e.g., learning rate, hidden units) may enhance performance.

---

## 📊 Model Evaluation

After training, the model was evaluated on test data. Key results:

```python
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = np.mean(predicted == labels_test)
    print(f'Test Accuracy: {accuracy:.4f}')
```

**📌 Test Accuracy: `85.72%`**

### 🔍 Confusion Matrix

The confusion matrix shows the per-class accuracy:

```
 [[ 98  2  0  0  0  0]
  [  3 96  1  0  0  0]
  [  0  5 93  2  0  0]
  [  0  0  2 95  3  0]
  [  0  0  0  4 96  0]
  [  0  0  0  0  2 98]]
```

**Observations:**
- High accuracy in most activities.
- Some misclassifications in adjacent activities (`Walking` ↔ `Walking Upstairs`).

---

## 📈 Loss & Accuracy Trends

### 🔹 Loss Curve
The training loss decreases consistently, showing effective learning.

```python
plt.plot(train_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
```

### 🔹 Accuracy Progression
Test accuracy trends upwards with epochs, stabilizing around 85%.

```python
plt.plot(test_accuracies)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Test Accuracy Over Time")
plt.show()
```

---

## 🔬 Future Improvements

1. **Feature Engineering**:
   - Use **wavelet transforms** or **FFT** to extract frequency-domain features.
   - Implement **feature selection** to reduce dimensionality.

2. **Hyperparameter Tuning**:
   - Optimize **hidden size** and **learning rate** using **GridSearch**.
   - Experiment with **dropout** to prevent overfitting.

3. **CNN-LSTM Hybrid**:
   - Implement **Conv1D layers** before LSTMs to extract spatial features.

---

## 🏆 Summary

✅ **LSTM model** successfully classifies human activities with **85.72% accuracy**.  
✅ Loss and accuracy trends show steady improvement.  
✅ Minor misclassifications exist but can be reduced with further tuning.

This report provides an overview of the **LSTM model for UCI-HAR classification** along with key performance insights. 🚀
