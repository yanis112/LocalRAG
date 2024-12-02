import json

import matplotlib.pyplot as plt
from datasets import Dataset
from dotenv import load_dotenv
from scipy.special import softmax
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from functools import lru_cache

from src.main_utils.utils import translate_to_english
from src.aux_utils.logging_utils import log_execution_time


load_dotenv()


def fine_tune_model(sentences, labels):
    # Create a dataset from the input lists
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.1, random_state=42
    )
    train_data = {"sentence": train_sentences, "label": train_labels}
    val_data = {"sentence": val_sentences, "label": val_labels}

    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    # Load a SetFit model from Hub
    model = SetFitModel.from_pretrained(
        'all-MiniLM-L6-v2', labels=["lexical", "semantic","very_lexical","very_semantic"]  #use_differentiable_head=True, head_params={"temperature": 2.0,"out_features": 2}
    )
    
    # Initially set the model head to a simple logistic regression
    base_classifier = LogisticRegression(class_weight="balanced",C=1,multi_class="multinomial",solver="lbfgs")
    #base_classifier = MLPClassifier(activation='relu',hidden_layer_sizes=(30,))
    # #calibrator = CalibratedClassifierCV(base_estimator=base_classifier, method='isotonic', cv='prefit')
    model.model_head = base_classifier
    
    #print("Initial MODEL HEAD:", model.model_head)

    args = TrainingArguments(
        batch_size=4,
        num_epochs=2, #3 avant
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        metric="accuracy",
        column_mapping={
            "sentence": "text",
            "label": "label",
        },
    )

    
    
    # Train and evaluate with logistic regression head
    trainer.train()
    metrics = trainer.evaluate(val_dataset)
    print("Evaluation metrics:", metrics)

    # After training, retrieve the model from the trainer
    trained_model = trainer.model
    # print("TRAINED MDOEL OBJECT:", trained_model)
    # print("TRAINED MODEL HEAD:", trained_model.model_head)
    # print("Type of trained model head:", type(trained_model.model_head))
    logistic_head = trained_model.model_head

    # Create a calibrator for the model with the trained logistic regression head
    #calibrator = CalibratedClassifierCV(base_estimator=logistic_head, method='isotonic', cv=2)
    # # Fit the calibrator on the validation dataset
    # print("Nombre d'exemples dans le dataset de validation:", len(val_sentences))
    # print("Nombre d'exemples dans le dataset de validation pour le label lexical:", len([label for label in val_labels if label == "lexical"]))
    # print("Nombre d'exemples dans le dataset de validation pour le label semantic:", len([label for label in val_labels if label == "semantic"]))
    
    #print("Built calibrator:", calibrator)
    #fitted_callibrator=calibrator.fit(X=val_sentences, y=val_labels)
    
    #print("Fitted calibrator:", fitted_callibrator)

    # # Change the head of the model to the calibrated classifier
    #trained_model.model_head = calibrator
    # print("New MODEL HEAD:", trained_model.model_head)

    # Plotting the training and validation loss
    training_stats = trainer.state.log_history
    print("training_stats:", training_stats)
    train_loss = [x["embedding_loss"] for x in training_stats if "embedding_loss" in x]
    eval_loss = [x["eval_embedding_loss"] for x in training_stats if "eval_embedding_loss" in x]
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="train_loss")
    plt.plot(eval_loss, label="eval_loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("src/loss_plot.png")
    plt.show()

    
    # Push the model with the calibrated head to the Hub
    trainer.push_to_hub("yaniseuranova/setfit-rag-hybrid-search-query-router-test")

    # Download the updated model from the Hub
    model = SetFitModel.from_pretrained("yaniseuranova/setfit-rag-hybrid-search-query-router-test")
    return model




def plot_output_distrib():
    # Load the trained model
    model = SetFitModel.from_pretrained("yaniseuranova/setfit-rag-hybrid-search-query-router-test")

    # Load the dataset
    with open("src/datasets/query_router_finetuning_dataset_advanced.json", "r") as f:
        dataset = json.load(f)

    sentences = [dataset[key]["query"] for key in dataset]
    true_labels = [dataset[key]["label"] for key in dataset]  # Assuming there's a 'label' key for true labels

    # Initialize lists to store probabilities
    lexical_probs = []
    semantic_probs = []
    predicted_probs = []  # For calibration curve

    # Iterate over sentences to predict and store probabilities
    for sentence, true_label in zip(sentences, true_labels):
        # Translate the sentence to English (assuming the model expects English input)
        sentence = translate_to_english(sentence)
        # Get prediction probabilities
        preds = model.predict_proba([sentence])
        probas =preds[0].tolist() 
        #print("PROBAS:", probas)
        
        
        #probas = softmax(preds[0]).tolist()  # Apply softmax to get probabilities
        # Assuming the first label is "lexical" and the second is "semantic"
        lexical_probs.append(probas[0])
        semantic_probs.append(probas[1])
        
        if true_label == "lexical":
            predicted_probs.append(probas[0])
        elif true_label == "semantic":
            predicted_probs.append(probas[1])

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    # Distribution of probabilities for "lexical"
    axs[0].hist(lexical_probs, bins=20, color='skyblue')
    axs[0].set_title('Distribution of Probabilities for "lexical"')
    axs[0].set_xlabel('Probability')
    axs[0].set_ylabel('Frequency')

    # Distribution of probabilities for "semantic"
    axs[1].hist(semantic_probs, bins=20, color='salmon')
    axs[1].set_title('Distribution of Probabilities for "semantic"')
    axs[1].set_xlabel('Probability')
    axs[1].set_ylabel('Frequency')
        
        
    # Convert string labels to numerical labels
    numerical_true_labels = [0 if label == "lexical" else 1 for label in true_labels]

    # Calibration curve
    prob_true, prob_pred = calibration_curve(numerical_true_labels, predicted_probs, n_bins=10)
    axs[2].plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    axs[2].plot(prob_pred, prob_true, marker='.')
    axs[2].set_title('Calibration curve')
    axs[2].set_xlabel('Mean predicted probability')
    axs[2].set_ylabel('Fraction of positives')

    plt.tight_layout()
    plt.savefig("probabilities_distribution_and_calibration.png")
    plt.show()

class QueryRouter:
    def __init__(self):
        self.model = None

    @log_execution_time
    @lru_cache(maxsize=None)
    def load(self):
        self.model = SetFitModel.from_pretrained(
            "yaniseuranova/setfit-rag-hybrid-search-query-router"
        )

    @log_execution_time
    def predict_label(self, sentence):
        # we first need to translate the sentence to english
        sentence = translate_to_english(sentence)

        if self.model is None:
            raise Exception(
                "Model not loaded. Please call the 'load' method before 'predict'."
            )

        preds = self.model.predict([sentence],batch_size=1)

        # Assuming the model returns a list of labels, we take the first one
        return preds[0]

    @log_execution_time
    def predict_alpha(self, sentence, temperature=1.5):
        preds = self.model.predict_proba([sentence])
        probas = preds[0].tolist()
        #print("PROBAS:", probas)
        # Divide each probability by the temperature
        probas = [p / temperature for p in probas]
        # Apply softmax
        probas = softmax(probas)
        # We get one probability for each label, if max prob is the first label, we return alpha=prob[0] else we return alpha=1-prob[0]
        if probas[0] > probas[1]:
            return probas[0]
        else:
            return 1 - probas[1]
        
        
def calibrate_model_temperature_scaling(model):
    """
    Calibrates the trained model using temperature scaling on a softmax function.
    Temperature scaling is a post-processing step to adjust the confidence of the model's predictions.

    Parameters:
    - model: The trained SetFitModel instance.
    - val_dataset: The validation dataset used for calibration.

    Returns:
    - The model with its logits adjusted by the learned temperature.
    """

    import numpy as np
    from scipy.optimize import minimize
    
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    sentences, labels, test_size=0.1, random_state=42
)
    train_data = {"sentence": train_sentences, "label": train_labels}
    val_data = {"sentence": val_sentences, "label": val_labels}

    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    def softmax_temperature(logits, temperature):
        """
        Applies softmax function with temperature scaling.
        """
        e_x = np.exp((logits - np.max(logits)) / temperature)
        return e_x / e_x.sum(axis=1, keepdims=True)

    def nll_loss(temperature, logits, labels):
        """
        Computes the Negative Log Likelihood loss given a temperature.
        """
        scaled_logits = softmax_temperature(logits, temperature)
        return -np.mean(np.log(scaled_logits[np.arange(len(labels)), labels]))

    def find_optimal_temperature(logits, labels):
        """
        Finds the optimal temperature using the NLL loss as the objective function.
        """
        result = minimize(nll_loss, x0=1.0, args=(logits, labels), bounds=[(0.01, 5.0)])
        return result.x[0]

    # Extract logits and labels from the validation dataset
    logits = []
    labels = []
    for example in val_dataset:
        logits.append(model.predict([example['sentence']])[0])
        labels.append(example['label'])

    logits = np.array(logits)
    labels = np.array(labels)

    # Find the optimal temperature
    optimal_temperature = find_optimal_temperature(logits, labels)

    # Adjust the model's logits using the learned temperature
    def adjust_logits_with_temperature(model_logits):
        return softmax_temperature(model_logits, optimal_temperature)

    # Assuming the model has a method to adjust logits or a way to apply a custom function to its output
    # This part is pseudo-code and needs to be adapted to the actual model implementation
    model.adjust_logits = adjust_logits_with_temperature

    return model


if __name__ == "__main__":
    # Load data from JSON file
 
    
    
    
    
    
    #exit()
    with open(
        "src/datasets/query_router_finetuning_dataset_4_claases.json","r"                  #advanced.json", "r"   #advanced dataset !!!!
    ) as f:  # 'src/few_shot_classif_dataset.json'
        dataset = json.load(f)

    sentences = [dataset[key]["query"] for key in dataset]
    labels = [dataset[key]["label"] for key in dataset]

    model = fine_tune_model(sentences, labels)
    
    plot_output_distrib()
    
    exit()
    
    
    #calibrate 
    # model = calibrate_model_temperature_scaling(model)
    

    # exit()

    # Run inference
    list_test = [
        "Quels sont les avantages de l'utilisation de l'intelligence artificielle?",
        "Quel est le prix actuel du Bitcoin?",
        "Comment améliorer la productivité des employés?",
        "Quelle est la hauteur de la Tour Eiffel?",
        "Comment réduire sa consommation de RAM gpu ?",
        "Qu'est ce que sont les réacteurs nucléaires à neutrons rapides et a l'uranium enrichi ?",
        "Qui est Quentin ?",
        "Qu'est ce que la machine Vidar ?",
    ]

    query_router = QueryRouter()
    query_router.load()
    for test in list_test:
        print(f"Query: {test}")
        print(f"Prediction: {query_router.predict_label(test)}")
        print(f"Alpha: {query_router.predict_alpha(test)}")

# %%
