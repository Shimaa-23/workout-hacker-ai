class RestRecommender:

    def __init__(self):
        self.history = []

    def update(self, fatigue_result):

        level = fatigue_result["fatigue_level"]

        self.history.append(level)

        # keep last 3 readings only
        if len(self.history) > 3:
            self.history.pop(0)

        # majority vote smoothing
        from collections import Counter
        stable_level = Counter(self.history).most_common(1)[0][0]

        if stable_level == "low":
            return {"action": "continue", "message": "You're good", "rest": False}

        elif stable_level == "moderate":
            return {"action": "warning", "message": "Slow down", "rest": False}

        elif stable_level == "high":
            return {"action": "rest_suggested", "message": "Take a break", "rest": True}

        else:
            return {"action": "rest_required", "message": "Stop now", "rest": True}