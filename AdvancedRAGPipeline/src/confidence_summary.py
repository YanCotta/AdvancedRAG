from typing import Dict, Tuple
import numpy as np

class ConfidenceSummary:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        
    def analyze_response(self, text: str, confidence: float) -> Dict:
        """Analyzes response and generates confidence metrics."""
        result = {
            "text": text,
            "confidence": confidence,
            "requires_review": confidence < self.confidence_threshold,
            "uncertain_segments": []
        }
        
        # Identify potentially uncertain segments
        sentences = text.split(". ")
        for sentence in sentences:
            if any(word in sentence.lower() for word in 
                  ["maybe", "might", "could", "possibly", "perhaps"]):
                result["uncertain_segments"].append(sentence)
                
        return result

    def format_output(self, analysis: Dict) -> str:
        """Formats the analysis into a user-friendly output."""
        output = f"Answer (Confidence: {analysis['confidence']:.2f}):\n"
        output += analysis['text'] + "\n"
        
        if analysis['requires_review']:
            output += "\n⚠️ This response may need human review.\n"
            
        if analysis['uncertain_segments']:
            output += "\nUncertain segments:\n"
            for segment in analysis['uncertain_segments']:
                output += f"- {segment}\n"
                
        return output
