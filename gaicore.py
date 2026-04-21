import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier

class GAICore:
    def __init__(self):
        self.memory = {}
        self.trained = False

        self.models = {
            "neural_network": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000),
            "decision_tree": DecisionTreeClassifier()
        }

    def _ensure_trained(self, input_dim):
        """Auto-train models with dummy data if not trained yet."""
        if not self.trained:
            X = np.random.rand(50, input_dim)
            y = np.random.rand(50)

            self.models["neural_network"].fit(X, y)
            self.models["decision_tree"].fit(X, (y > 0.5).astype(int))

            self.trained = True

    def perceive(self, environment_data):
        perception = {
            "patterns": np.mean(environment_data, axis=0),
            "outliers": [
                data for data in environment_data
                if np.linalg.norm(data) > 2 * np.mean(environment_data)
            ]
        }
        return perception

    def reason(self, input_data):
        self._ensure_trained(len(input_data))

        if random.choice([True, False]):
            decision = self.models["decision_tree"].predict([input_data])[0]
            reasoning = f"Logical reasoning: Selected class {decision}."
        else:
            decision = self.models["neural_network"].predict([input_data])[0]
            reasoning = f"Neural reasoning: Predicted value {round(float(decision), 4)}."

        return reasoning, decision

    def learn(self, inputs, outputs):
        self.models["neural_network"].fit(inputs, outputs)
        self.models["decision_tree"].fit(inputs, (outputs > 0.5).astype(int))
        self.trained = True
        return "Learning complete."

    def adapt(self, feedback):
        success = feedback.get("success", False)
        key = "positive_feedback" if success else "negative_feedback"
        self.memory[key] = feedback
        return f"Adapted to feedback: {feedback}"

    def interact(self, input_data, feedback=None):
        perception = self.perceive(input_data)
        reasoning, decision = self.reason(perception["patterns"])

        response = {
            "perception": perception,
            "reasoning": reasoning,
            "decision": float(decision) if not isinstance(decision, int) else decision
        }

        if feedback:
            self.adapt(feedback)

        return response


if __name__ == "__main__":
    gai = GAICore()
    data = np.random.rand(100, 10)

    print(gai.interact(data))
