def evaluate_using_mnemonics(patient_data, model_output):
    """
    Evaluate the patient case using clinical mnemonics such as ASMETHOD, WWHAM, ENCORE, and SIT DOWN SIR.
    
    Parameters:
      patient_data: dict with keys (e.g., "age", "symptoms", "time_persisting", etc.)
      model_output: probability vector from the TensorFlow model (e.g., [otc, refer, further_eval])
    
    Returns:
      A decision string (e.g., "Recommend OTC medication", "Refer to doctor", etc.)
    """
    # Example rule: if any danger symptoms are present, refer immediately.
    symptoms = patient_data.get("symptoms", "").lower()
    danger_keywords = ["chest pain", "difficulty breathing", "severe", "unconscious"]
    if any(keyword in symptoms for keyword in danger_keywords):
        return "Refer to doctor immediately"

    # Example rule based on ASMETHOD: if symptoms persist for more than 7 days.
    if patient_data.get("time_persisting", 0) > 7:
        return "Further evaluation required due to prolonged symptoms"

    # Combine heuristic with model prediction:
    # Here we assume the model output is a probability vector in the order:
    # [OTC_recommendation, refer_to_doctor, further_evaluation]
    otc_confidence = model_output[0]
    refer_confidence = model_output[1]
    further_eval_confidence = model_output[2]

    if refer_confidence > 0.5:
        return "Refer to doctor for further evaluation"
    elif otc_confidence > 0.5:
        return "Recommend OTC medication"
    else:
        return "Further evaluation required"
