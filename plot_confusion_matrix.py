from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming 'y_true' contains true labels and 'y_pred' contains predicted labels
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
