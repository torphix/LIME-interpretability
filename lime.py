import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from pysentimiento import create_analyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer




def tokenize(text):
   return text.split()


def create_perturbed_samples(text, num_samples):
   tokens = tokenize(text)
   perturbed_samples = []
   for _ in range(num_samples):
       selected_indices = np.random.choice(range(len(tokens)), size=int(len(tokens) * 0.75), replace=False)
       perturbed_text = ' '.join([tokens[i] for i in selected_indices])
       perturbed_samples.append(perturbed_text)
   return perturbed_samples


def lime_text_explainer(text, model, num_samples=100, num_features=10):
   # Create perturbed samples
   perturbed_samples = create_perturbed_samples(text, num_samples)
  
   # Vectorize samples
   vectorizer = CountVectorizer(tokenizer=tokenize)
   perturbed_vectors = vectorizer.fit_transform(perturbed_samples).toarray()
   original_vector = vectorizer.transform([text]).toarray()
  
   # Compute sample weights based on similarity
   sample_weights = cosine_similarity(original_vector, perturbed_vectors).flatten()
  
   # Get model predictions for perturbed samples
   analyzer_outputs = [model.predict(preprocessed_text) for preprocessed_text in perturbed_samples]
   predictions = np.array([[output.probas[class_name] for class_name in analyzer_outputs[0].probas] for output in analyzer_outputs])
  
   # Train interpretable models for each class
   feature_importances = []
   num_classes = predictions.shape[1]
   for class_idx in range(num_classes):
       ridge_model = Ridge(alpha=1, fit_intercept=True)
       ridge_model.fit(perturbed_vectors, predictions[:, class_idx], sample_weight=sample_weights)


       # Get top feature importances
       top_features = np.argsort(np.abs(ridge_model.coef_))[-num_features:]
       top_feature_importances = ridge_model.coef_[top_features]


       feature_importances.append(list(zip(top_features, top_feature_importances)))
  
   return feature_importances, vectorizer


def run_lime(text, model):
   # Get feature importances for all classes
   feature_importances, vectorizer = lime_text_explainer(text, model)


   # Print feature importances and plot the results
   emotion_names = list(model.predict(text).probas.keys())
   output_class = model.predict(text).output
   print(emotion_names)
   for class_idx, importances in enumerate(feature_importances):
       if output_class == emotion_names[class_idx]:
           print(f"{emotion_names[class_idx]} feature importances:")
           feature_names = []
           importance_values = []
           for feature, importance in importances:
               feature_name = vectorizer.get_feature_names_out()[feature]
               print(f"  Feature {feature_name}: {importance}")
               feature_names.append(feature_name)
               importance_values.append(importance)
      
           # Plot feature importances for the current class
           plot_feature_importances(importance_values, feature_names, emotion_names[class_idx])




def plot_feature_importances(importances, feature_names, class_name):
   indices = np.arange(len(importances))
   plt.bar(indices, importances, align='center')
   plt.xticks(indices, feature_names, rotation=45, ha='right')
   plt.xlabel('Features')
   plt.ylabel('Importance')
   plt.title(f'{class_name} Feature Importances')
   plt.tight_layout()
   plt.show()


emotion_analyzer = create_analyzer(task="emotion", lang="en")
run_lime('my heart is broken into a thousand pieces', emotion_analyzer)
run_lime('todays was the best day of my life', emotion_analyzer)
run_lime('that is so gross, i might vomit', emotion_analyzer)
run_lime('i could not believe what she told me, i was in utter shock', emotion_analyzer)
run_lime('I was so scared, i had to run away', emotion_analyzer)


