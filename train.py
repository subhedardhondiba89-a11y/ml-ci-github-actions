from src.model import train_model

def main():
    model, accuracy = train_model()
    print("ML Model Training Completed!")
    print(f"Model: {type(model).__name__}")
    print(f"Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
