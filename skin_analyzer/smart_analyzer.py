#!/usr/bin/env python3
"""
Smart Skin Analyzer - Optimized for Real Photos
Uses Random Forest as default with intelligent result interpretation
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_unified import UnifiedSkinTypePredictor

def smart_skin_analysis(image_path, save_report=False):
    """
    Perform intelligent skin analysis optimized for real photos
    """
    print("🔬 SMART SKIN ANALYZER - OPTIMIZED FOR REAL PHOTOS")
    print("=" * 60)
    print(f"📸 Analyzing: {os.path.basename(image_path)}")
    print("🌳 Using Random Forest (best for real photos)")
    print("-" * 60)
    
    # Use Random Forest as default (best for real photos)
    predictor = UnifiedSkinTypePredictor(model_type='random_forest')
    result = predictor.analyze_skin_characteristics(image_path)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Extract key information
    skin_type = result['skin_type']
    confidence = result['confidence']
    probabilities = result['probabilities']
    face_detected = result.get('face_detected', False)
    
    # Smart interpretation
    print(f"\n🎯 PRIMARY PREDICTION:")
    print(f"   Skin Type: {skin_type.upper()}")
    print(f"   Confidence: {confidence:.1%}")
    print(f"   Face Detected: {'✅ Yes' if face_detected else '❌ No'}")
    
    # Intelligent confidence interpretation
    print(f"\n🧠 CONFIDENCE INTERPRETATION:")
    if confidence >= 0.6:
        print(f"   🟢 Relatively clear characteristics")
        print(f"   📝 Primary skin type is likely {skin_type}")
    elif confidence >= 0.4:
        print(f"   🟡 Mixed characteristics detected")
        print(f"   📝 Leaning toward {skin_type} but with variations")
    else:
        print(f"   🟠 Complex/Combination skin detected")
        print(f"   📝 Multiple skin types present - needs nuanced care")
    
    # Show probability distribution with smart grouping
    print(f"\n📊 SKIN TYPE BREAKDOWN:")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    primary_types = []
    secondary_types = []
    
    for skin_type_name, prob in sorted_probs:
        if prob >= 0.15:  # 15% threshold for significant characteristics
            primary_types.append((skin_type_name, prob))
        elif prob >= 0.05:  # 5% threshold for minor characteristics
            secondary_types.append((skin_type_name, prob))
    
    print(f"   Primary Characteristics:")
    for skin_type_name, prob in primary_types:
        bar = "█" * int(prob * 30)
        print(f"     {skin_type_name.capitalize():12}: {prob:.1%} {bar}")
    
    if secondary_types:
        print(f"   Minor Characteristics:")
        for skin_type_name, prob in secondary_types:
            print(f"     {skin_type_name.capitalize():12}: {prob:.1%}")
    
    # Smart skincare recommendations
    print(f"\n💡 SMART SKINCARE RECOMMENDATIONS:")
    
    if len(primary_types) == 1:
        # Single dominant type
        print(f"   🎯 FOCUSED APPROACH for {primary_types[0][0]} skin:")
        analysis = result['detailed_analysis']
        print(f"   • {analysis['care_tips'][0]}")
        print(f"   • {analysis['care_tips'][1]}")
    
    elif len(primary_types) == 2:
        # Two dominant types - combination approach
        type1, prob1 = primary_types[0]
        type2, prob2 = primary_types[1]
        print(f"   ⚖️  BALANCED APPROACH for {type1}/{type2} combination:")
        print(f"   • Use gentle products suitable for mixed skin")
        print(f"   • Target different areas with specific treatments")
        print(f"   • Monitor seasonal changes in skin behavior")
    
    else:
        # Complex skin
        print(f"   🎭 COMPLEX SKIN APPROACH:")
        print(f"   • Start with gentle, minimal routine")
        print(f"   • Introduce products gradually")
        print(f"   • Consider professional consultation")
    
    # Risk assessment
    print(f"\n⚠️  RISK ASSESSMENT:")
    if confidence < 0.3:
        print(f"   🟠 High uncertainty - consider multiple product testing")
        print(f"   📝 Professional dermatologist consultation recommended")
    elif len(primary_types) >= 3:
        print(f"   🟡 Complex skin - patch test new products")
        print(f"   📝 Gradual introduction of skincare changes")
    else:
        print(f"   🟢 Relatively straightforward skin type")
        print(f"   📝 Standard skincare routine should work well")
    
    # Save report if requested
    if save_report:
        report_file = f"smart_analysis_{os.path.splitext(os.path.basename(image_path))[0]}.json"
        predictor.save_prediction_report(result, report_file)
        print(f"\n📄 Detailed report saved: {report_file}")
    
    print(f"\n✨ Analysis complete! Random Forest provides the most realistic")
    print(f"   assessment for real photos by honestly showing uncertainty.")

def compare_with_other_models(image_path):
    """Compare Random Forest with other models to show the difference"""
    
    print(f"\n🔄 COMPARISON WITH OTHER MODELS (for reference):")
    print("-" * 50)
    
    models = [
        ('gradient_boost', '🚀'),
        ('svm', '🎯')
    ]
    
    rf_predictor = UnifiedSkinTypePredictor(model_type='random_forest')
    rf_result = rf_predictor.predict_image(image_path)
    
    print(f"🌳 Random Forest: {rf_result['skin_type'].upper()} ({rf_result['confidence']:.1%})")
    
    for model_type, emoji in models:
        try:
            predictor = UnifiedSkinTypePredictor(model_type=model_type)
            result = predictor.predict_image(image_path)
            
            if 'error' not in result:
                print(f"{emoji} {model_type.replace('_', ' ').title()}: "
                      f"{result['skin_type'].upper()} ({result['confidence']:.1%})")
            
        except Exception as e:
            print(f"{emoji} {model_type}: Error - {str(e)}")
    
    print(f"\n💭 Notice how Random Forest often shows more balanced probabilities")
    print(f"   and reasonable confidence levels compared to other models.")

def main():
    parser = argparse.ArgumentParser(description='Smart Skin Analyzer - Optimized for Real Photos')
    parser.add_argument('--image', '-i', required=True, help='Path to image file')
    parser.add_argument('--save-report', '-s', action='store_true', help='Save detailed JSON report')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare with other models')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        return
    
    # Main analysis
    smart_skin_analysis(args.image, args.save_report)
    
    # Optional comparison
    if args.compare:
        compare_with_other_models(args.image)

if __name__ == "__main__":
    main()
