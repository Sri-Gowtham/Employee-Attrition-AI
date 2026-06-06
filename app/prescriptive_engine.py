def generate_recommendations(employee_data, prob_quit, shap_df=None):
    """
    Generate retention recommendations based on employee features and risk score.
    Returns:
        dict containing recommendations categorized by timeframe and severity.
    """
    recs = []

    # Satisfaction
    satisfaction = employee_data.get('satisfaction_level', 1.0)
    if satisfaction < 0.4:
        recs.append({
            "action": "Employee Engagement Discussion",
            "reason": f"Satisfaction level is very low ({satisfaction}).",
            "severity": "Critical",
            "timeframe": "Immediate"
        })
    elif satisfaction < 0.6:
        recs.append({
            "action": "Satisfaction Check-in",
            "reason": f"Satisfaction level is below average ({satisfaction}).",
            "severity": "Medium",
            "timeframe": "Short-Term"
        })

    # Salary
    salary = employee_data.get('salary', 'high')
    if salary == 'low':
        recs.append({
            "action": "Salary Review",
            "reason": "Employee is in the low salary bracket.",
            "severity": "High",
            "timeframe": "Short-Term"
        })

    # Promotion
    promotion = employee_data.get('promotion_last_5years', 1)
    if promotion == 0:
        years = employee_data.get('time_spend_company', 0)
        if years >= 3:
            recs.append({
                "action": "Promotion Evaluation",
                "reason": f"No promotion in the last 5 years despite {years} years of tenure.",
                "severity": "High",
                "timeframe": "Short-Term"
            })
        else:
            recs.append({
                "action": "Career Path Mapping",
                "reason": "No promotion yet. Outline clear career progression.",
                "severity": "Medium",
                "timeframe": "Long-Term"
            })

    # Workload (Hours)
    hours = employee_data.get('average_montly_hours', 150)
    if hours > 250:
        recs.append({
            "action": "Workload Reduction",
            "reason": f"Working excessive hours ({hours}/month). High burnout risk.",
            "severity": "Critical",
            "timeframe": "Immediate"
        })
    elif hours < 130:
        recs.append({
            "action": "Role Utilization Review",
            "reason": f"Low monthly hours ({hours}/month). Might be disengaged or underutilized.",
            "severity": "Low",
            "timeframe": "Long-Term"
        })

    # Projects
    projects = employee_data.get('number_project', 3)
    if projects > 6:
        recs.append({
            "action": "Project Redistribution",
            "reason": f"Handling too many projects ({projects}). Risk of overload.",
            "severity": "High",
            "timeframe": "Short-Term"
        })
    elif projects < 3:
        recs.append({
            "action": "Skill/Project Alignment",
            "reason": f"Low project count ({projects}). Discuss opportunities for new challenges.",
            "severity": "Medium",
            "timeframe": "Long-Term"
        })

    # Ensure there is at least one generic recommendation if risk is high but rules didn't trigger
    if prob_quit > 0.5 and not recs:
        recs.append({
            "action": "General Retention Interview",
            "reason": "High attrition risk detected without obvious drivers. Perform an open-ended interview.",
            "severity": "High",
            "timeframe": "Immediate"
        })

    # Grouping
    immediate = [r for r in recs if r['timeframe'] == 'Immediate']
    short_term = [r for r in recs if r['timeframe'] == 'Short-Term']
    long_term = [r for r in recs if r['timeframe'] == 'Long-Term']

    # Estimate impact
    # Heuristic: Critical actions reduce risk by 15%, High by 10%, Medium by 5%, Low by 2%
    # Floor at 5%
    reduction = 0
    for r in recs:
        if r['severity'] == 'Critical': reduction += 0.15
        elif r['severity'] == 'High': reduction += 0.10
        elif r['severity'] == 'Medium': reduction += 0.05
        elif r['severity'] == 'Low': reduction += 0.02
    
    new_risk = max(0.05, prob_quit - reduction)
    if new_risk > prob_quit:
        new_risk = prob_quit

    return {
        "immediate": immediate,
        "short_term": short_term,
        "long_term": long_term,
        "all_recommendations": sorted(recs, key=lambda x: ['Critical', 'High', 'Medium', 'Low'].index(x['severity'])),
        "current_risk": prob_quit,
        "estimated_new_risk": new_risk
    }

def format_report_markdown(recs_result, employee_data):
    """
    Format the recommendations into a Markdown string for download.
    """
    md = "# Employee Retention Strategy Report\n\n"
    
    md += f"**Current Attrition Risk:** {recs_result['current_risk']:.1%}\n"
    md += f"**Estimated Risk after Actions:** {recs_result['estimated_new_risk']:.1%}\n\n"
    
    md += "## 🚨 Immediate Actions\n"
    if not recs_result['immediate']:
        md += "- None required.\n"
    for r in recs_result['immediate']:
        md += f"- **{r['action']}** (Severity: {r['severity']})\n  - *Reason:* {r['reason']}\n"
        
    md += "\n## 📅 Short-Term Actions\n"
    if not recs_result['short_term']:
        md += "- None required.\n"
    for r in recs_result['short_term']:
        md += f"- **{r['action']}** (Severity: {r['severity']})\n  - *Reason:* {r['reason']}\n"
        
    md += "\n## 📈 Long-Term Actions\n"
    if not recs_result['long_term']:
        md += "- None required.\n"
    for r in recs_result['long_term']:
        md += f"- **{r['action']}** (Severity: {r['severity']})\n  - *Reason:* {r['reason']}\n"
        
    return md
