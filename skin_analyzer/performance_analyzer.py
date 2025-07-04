#!/usr/bin/env python3
"""
Simplified Performance Analysis and Improvement Plan Generator
============================================================

Creates a comprehensive analysis of current model performance and provides
actionable improvement recommendations.
"""

import os
import json
import pandas as pd
from pathlib import Path

def analyze_current_performance():
    """Analyze current model performance and create improvement plan."""
    
    base_path = Path("/Users/bjit/Desktop/My_Files/Projects/Image_Analyzer/skin_analyzer")
    reports_path = base_path / "trained_models" / "reports"
    
    print("🚀 SKIN ANALYZER PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Load classification report
    report_path = reports_path / "classification_report.json"
    
    if not report_path.exists():
        print("❌ Classification report not found!")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("📊 CURRENT MODEL PERFORMANCE:")
    print(f"   Overall Accuracy: {report['accuracy']:.1%}")
    print()
    
    # Analyze per-class performance
    classes = ['combination', 'dry', 'normal', 'oily', 'sensitive']
    class_performance = []
    
    print("📈 PER-CLASS ANALYSIS:")
    for class_name in classes:
        if class_name in report:
            metrics = report[class_name]
            precision = metrics['precision']
            recall = metrics['recall']
            f1_score = metrics['f1-score']
            support = int(metrics['support'])
            
            # Determine performance level
            if f1_score >= 0.75:
                status = "🟢 EXCELLENT"
            elif f1_score >= 0.65:
                status = "🟡 GOOD"
            elif f1_score >= 0.55:
                status = "🟠 MODERATE"
            else:
                status = "🔴 NEEDS IMPROVEMENT"
            
            print(f"   {class_name.capitalize():12} | F1: {f1_score:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | {status}")
            
            class_performance.append({
                'class': class_name,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall,
                'support': support,
                'needs_improvement': f1_score < 0.65
            })
    
    # Identify priority areas
    priority_classes = [cp['class'] for cp in class_performance if cp['needs_improvement']]
    
    print("\n🎯 PRIORITY IMPROVEMENT AREAS:")
    if priority_classes:
        for class_name in priority_classes:
            class_data = next(cp for cp in class_performance if cp['class'] == class_name)
            print(f"   • {class_name.capitalize()}: F1-Score {class_data['f1_score']:.3f} (Target: 0.75+)")
    else:
        print("   🎉 All classes performing well!")
    
    # Create comprehensive improvement plan
    improvement_plan = create_improvement_plan(report['accuracy'], class_performance, priority_classes)
    
    # Save results
    results_dir = base_path / "model_improvement"
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed analysis
    analysis_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'current_accuracy': report['accuracy'],
        'target_accuracy': 0.80,
        'class_performance': class_performance,
        'priority_classes': priority_classes,
        'improvement_plan': improvement_plan
    }
    
    with open(results_dir / "performance_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Create readable improvement plan
    create_readable_plan(results_dir, improvement_plan, report['accuracy'], priority_classes)
    
    print(f"\n📁 Results saved to: {results_dir}")
    print(f"📋 Read the full plan: {results_dir}/IMPROVEMENT_ROADMAP.md")
    
    return analysis_results

def create_improvement_plan(current_accuracy, class_performance, priority_classes):
    """Create a comprehensive improvement plan."""
    
    target_accuracy = 0.80
    improvement_needed = target_accuracy - current_accuracy
    
    strategies = []
    
    # Strategy 1: Data Collection
    if priority_classes:
        strategies.append({
            'name': 'Enhanced Data Collection',
            'priority': 'HIGH',
            'description': f'Collect 50-100 additional high-quality images for {", ".join(priority_classes)}',
            'actions': [
                'Use Pexels API for additional skin images',
                'Search dermatology image databases',
                'Focus on underperforming classes',
                'Ensure diverse lighting and demographics'
            ],
            'expected_improvement': '5-10%',
            'timeline': '1-2 weeks'
        })
    
    # Strategy 2: Feature Engineering
    if current_accuracy < 0.75:
        strategies.append({
            'name': 'Advanced Feature Engineering',
            'priority': 'HIGH',
            'description': 'Implement sophisticated image analysis techniques',
            'actions': [
                'Local Binary Patterns (LBP) for texture analysis',
                'Color histogram analysis in multiple color spaces',
                'Edge density and smoothness metrics',
                'Skin region detection and isolation'
            ],
            'expected_improvement': '8-15%',
            'timeline': '1 week'
        })
    
    # Strategy 3: Model Enhancement
    strategies.append({
        'name': 'Advanced Model Techniques',
        'priority': 'MEDIUM',
        'description': 'Implement state-of-the-art ML approaches',
        'actions': [
            'Ensemble methods with weighted voting',
            'Hyperparameter optimization with Bayesian search',
            'Cross-validation with stratified sampling',
            'Model stacking and blending'
        ],
        'expected_improvement': '3-8%',
        'timeline': '1 week'
    })
    
    # Strategy 4: Deep Learning
    if improvement_needed > 0.10:
        strategies.append({
            'name': 'Deep Learning Implementation',
            'priority': 'MEDIUM',
            'description': 'Deploy CNN models with transfer learning',
            'actions': [
                'Fine-tune ResNet50 on skin classification',
                'Implement EfficientNet for mobile deployment',
                'Use pre-trained models from medical imaging',
                'Data augmentation with advanced techniques'
            ],
            'expected_improvement': '15-25%',
            'timeline': '2-3 weeks'
        })
    
    # Strategy 5: Data Quality
    strategies.append({
        'name': 'Data Quality Enhancement',
        'priority': 'LOW',
        'description': 'Improve training data quality and consistency',
        'actions': [
            'Remove low-quality or mislabeled images',
            'Standardize image preprocessing pipeline',
            'Implement quality scoring for images',
            'Balance dataset across demographics'
        ],
        'expected_improvement': '2-5%',
        'timeline': '1 week'
    })
    
    return {
        'current_accuracy': current_accuracy,
        'target_accuracy': target_accuracy,
        'improvement_needed': improvement_needed,
        'strategies': strategies,
        'recommended_order': [s['name'] for s in strategies if s['priority'] == 'HIGH'] + 
                            [s['name'] for s in strategies if s['priority'] == 'MEDIUM']
    }

def create_readable_plan(results_dir, plan, current_accuracy, priority_classes):
    """Create a readable improvement roadmap."""
    
    content = f"""# 🚀 Skin Analyzer Improvement Roadmap

## Current Status
- **Current Accuracy**: {current_accuracy:.1%}
- **Target Accuracy**: {plan['target_accuracy']:.1%}
- **Improvement Needed**: {plan['improvement_needed']:.1%}

## Classes Requiring Attention
{chr(10).join([f"- **{class_name.capitalize()}**: Needs performance boost" for class_name in priority_classes]) if priority_classes else "✅ All classes performing adequately"}

---

## 🎯 Improvement Strategies

"""
    
    for i, strategy in enumerate(plan['strategies'], 1):
        priority_emoji = "🔥" if strategy['priority'] == 'HIGH' else "⚡" if strategy['priority'] == 'MEDIUM' else "📋"
        
        content += f"""### {i}. {strategy['name']} {priority_emoji} {strategy['priority']} PRIORITY

**Expected Improvement**: {strategy['expected_improvement']}  
**Timeline**: {strategy['timeline']}

**Description**: {strategy['description']}

**Action Items**:
{chr(10).join([f"- {action}" for action in strategy['actions']])}

---

"""
    
    content += f"""## 📅 Recommended Implementation Order

{chr(10).join([f"{i+1}. {strategy}" for i, strategy in enumerate(plan['recommended_order'])])}

## 🎯 Success Metrics
- **Target Overall Accuracy**: 80%+
- **Target Per-Class F1-Score**: 75%+ for all classes
- **Confidence Threshold**: 85%+ for predictions
- **Inference Speed**: <2 seconds per image

## 🔧 Next Steps
1. **Immediate (This Week)**: Implement enhanced feature engineering
2. **Short-term (1-2 Weeks)**: Collect additional training data for priority classes
3. **Medium-term (2-4 Weeks)**: Deploy advanced ML techniques and deep learning
4. **Long-term (1-2 Months)**: Production optimization and deployment

## 📊 Expected Results Timeline
- **Week 1**: 5-10% accuracy improvement from feature engineering
- **Week 2**: Additional 5-8% from enhanced data collection
- **Week 3-4**: Final 10-15% boost from advanced models
- **Final Target**: 80%+ accuracy with robust performance across all skin types

---

*Generated on {pd.Timestamp.now().strftime('%B %d, %Y at %I:%M %p')}*
"""
    
    with open(results_dir / "IMPROVEMENT_ROADMAP.md", 'w') as f:
        f.write(content)

def main():
    """Main execution function."""
    try:
        results = analyze_current_performance()
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("\n🎯 KEY FINDINGS:")
        if results['priority_classes']:
            print(f"   • {len(results['priority_classes'])} classes need improvement")
            print(f"   • Focus areas: {', '.join(results['priority_classes'])}")
        print(f"   • Potential improvement: {results['improvement_plan']['improvement_needed']:.1%}")
        
        print("\n📋 NEXT ACTIONS:")
        high_priority = [s for s in results['improvement_plan']['strategies'] if s['priority'] == 'HIGH']
        for strategy in high_priority[:2]:  # Show top 2 high priority items
            print(f"   • {strategy['name']}: {strategy['expected_improvement']} improvement")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Make sure the model training has been completed and reports exist.")

if __name__ == "__main__":
    main()
