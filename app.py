
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Iris Flower Recognition Project")
print("=" * 60)
print("Author: Dr. Muhammad Usman")
print("=" * 60)

if os.path.exists('iris_model.pkl') and os.path.exists('scaler.pkl') and os.path.exists('label_encoder.pkl'):
    print("\nLoading saved model...")
    final_model = joblib.load('iris_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("Model loaded successfully!")
else:
    
    try:
        df = pd.read_csv('iris.csv')
    except FileNotFoundError:
        print("Error: iris.csv not found!")
        print("Please make sure iris.csv is in the same folder as this script.")
        exit()
    
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    if 'Id' in df.columns:
        X = df.drop(['Id', 'Species'], axis=1)
    else:
        X = df.drop(['Species'], axis=1)
    
    y = df['Species']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    print(f"\nData Split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set: {X_test.shape[0]} samples")
    
    print("\n\n🤖 STEP 3: TRAINING THE MODEL")
    
    k_values = range(1, 16)
    accuracy_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    
    best_k = k_values[np.argmax(accuracy_scores)]
    best_accuracy = max(accuracy_scores)
    
    print(f"\nBest k value: {best_k} with {best_accuracy*100:.2f}% accuracy")
    
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train, y_train)
    print("\nFinal model trained successfully!")
    
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {accuracy*100:.2f}%")
    
    joblib.dump(final_model, 'iris_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("\nModel saved for future use!")

print("Enter measurements when prompted (press Ctrl+C to exit)\n")

interactive_mode = input("Start interactive mode? (y/n): ").lower()

if interactive_mode == 'y':
    try:
        while True:
            print("\n" + "-" * 30)
            
            try:
                sl = float(input("Enter Sepal Length (cm): "))
                sw = float(input("Enter Sepal Width (cm): "))
                pl = float(input("Enter Petal Length (cm): "))
                pw = float(input("Enter Petal Width (cm): "))
            except ValueError:
                print("Invalid input! Please enter numbers only.")
                continue
            
            new_flower = pd.DataFrame({
                'SepalLengthCm': [sl],
                'SepalWidthCm': [sw],
                'PetalLengthCm': [pl],
                'PetalWidthCm': [pw]
            })
            
            new_flower_scaled = scaler.transform(new_flower)
            prediction = final_model.predict(new_flower_scaled)[0]
            probability = final_model.predict_proba(new_flower_scaled)[0]
            
            species = label_encoder.inverse_transform([prediction])[0]
            confidence = probability[prediction] * 100
            
            print(f"\n🔮 Prediction: {species}")
            print(f"   Confidence: {confidence:.1f}%")
            
            print("\n   Probability distribution:")
            for j, sp in enumerate(label_encoder.classes_):
                prob = probability[j] * 100
                print(f"     {sp:12}: {prob:.1f}%")
            
            again = input("\nPredict another flower? (y/n): ").lower()
            if again != 'y':
                break
                
    except KeyboardInterrupt:
        print("\n\n👋 Exiting interactive mode...")

print("\n" + "=" * 60)
print("Project Completed Successfully!")
print("=" * 60)