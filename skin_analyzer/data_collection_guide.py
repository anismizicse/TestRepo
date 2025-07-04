#!/usr/bin/env python3
"""
Data Collection and Improvement Guide for Skin Type Classification
Comprehensive strategy for building a high-quality training dataset
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from collections import defaultdict

class SkinDatasetImprover:
    """
    Tools and guidelines for improving skin type classification dataset
    """
    
    def __init__(self, dataset_dir="improved_dataset"):
        self.dataset_dir = dataset_dir
        self.skin_types = {
            'normal': {
                'description': 'Balanced skin, not too oily or dry',
                'characteristics': ['Even texture', 'Small pores', 'No sensitivity'],
                'target_images': 500
            },
            'dry': {
                'description': 'Lacks moisture, may feel tight',
                'characteristics': ['Rough texture', 'Visible flaking', 'Tight feeling'],
                'target_images': 500
            },
            'oily': {
                'description': 'Excess sebum production',
                'characteristics': ['Shiny appearance', 'Large pores', 'Prone to acne'],
                'target_images': 500
            },
            'combination': {
                'description': 'Oily T-zone, normal/dry elsewhere',
                'characteristics': ['Oily forehead/nose', 'Normal cheeks', 'Mixed texture'],
                'target_images': 500
            },
            'sensitive': {
                'description': 'Reacts easily to products/environment',
                'characteristics': ['Redness', 'Irritation', 'Stinging sensation'],
                'target_images': 500
            }
        }
        
    def create_data_collection_guide(self):
        """Generate comprehensive data collection guidelines"""
        
        guide = {
            "SKIN TYPE CLASSIFICATION - DATA COLLECTION GUIDE": {
                "overview": "This guide helps you collect high-quality, diverse skin images for training accurate skin type classification models.",
                
                "data_requirements": {
                    "total_images_needed": "2500+ images (500 per skin type)",
                    "image_quality": "High resolution (min 512x512 pixels)",
                    "lighting": "Natural lighting preferred, varied conditions",
                    "angles": "Front-facing, 45-degree angles, close-ups",
                    "diversity": "Multiple ethnicities, ages, genders"
                },
                
                "collection_strategies": {
                    "1_medical_databases": {
                        "sources": [
                            "Dermatology atlases",
                            "Medical image databases",
                            "Research institution datasets",
                            "FDA-approved skin analysis apps"
                        ],
                        "advantages": "Professional quality, expert-labeled",
                        "considerations": "May require licensing, limited diversity"
                    },
                    
                    "2_volunteer_photography": {
                        "setup": "Controlled photography sessions",
                        "requirements": [
                            "Consistent lighting setup",
                            "Professional camera or high-end smartphone",
                            "Standardized angles and distances",
                            "Consent forms and privacy protection"
                        ],
                        "validation": "Dermatologist review and labeling"
                    },
                    
                    "3_synthetic_augmentation": {
                        "techniques": [
                            "GAN-generated skin textures",
                            "Style transfer from existing images",
                            "3D skin modeling and rendering",
                            "Physics-based skin simulation"
                        ],
                        "benefits": "Unlimited data, controlled variations",
                        "limitations": "May not capture real-world complexity"
                    },
                    
                    "4_crowdsourcing": {
                        "platforms": [
                            "Amazon Mechanical Turk",
                            "Appen",
                            "Clickworker",
                            "Custom mobile app"
                        ],
                        "quality_control": [
                            "Multiple annotators per image",
                            "Expert validation",
                            "Consistency checks",
                            "Rejection criteria"
                        ]
                    }
                },
                
                "labeling_guidelines": {
                    "normal_skin": {
                        "criteria": [
                            "Balanced oil/moisture levels",
                            "Even skin tone",
                            "Small, barely visible pores",
                            "No frequent breakouts",
                            "Smooth texture"
                        ],
                        "examples": "Well-balanced, healthy-looking skin"
                    },
                    
                    "dry_skin": {
                        "criteria": [
                            "Visible flaking or scaling",
                            "Rough, uneven texture",
                            "Tight feeling after cleansing",
                            "Fine lines more prominent",
                            "Dull appearance"
                        ],
                        "examples": "Skin that looks parched or flaky"
                    },
                    
                    "oily_skin": {
                        "criteria": [
                            "Shiny, greasy appearance",
                            "Large, visible pores",
                            "Frequent breakouts/blackheads",
                            "Thick skin texture",
                            "Makeup doesn't last long"
                        ],
                        "examples": "Visibly shiny, especially T-zone"
                    },
                    
                    "combination_skin": {
                        "criteria": [
                            "Oily T-zone (forehead, nose, chin)",
                            "Normal to dry cheeks",
                            "Mixed pore sizes",
                            "Breakouts mainly in T-zone",
                            "Different textures in different areas"
                        ],
                        "examples": "Clearly different skin characteristics in different facial areas"
                    },
                    
                    "sensitive_skin": {
                        "criteria": [
                            "Visible redness or irritation",
                            "Reactive to products/environment",
                            "Burning or stinging sensations",
                            "Prone to rashes",
                            "Thin skin appearance"
                        ],
                        "examples": "Skin showing signs of irritation or reactivity"
                    }
                },
                
                "image_specifications": {
                    "technical_requirements": {
                        "resolution": "Minimum 512x512 pixels, preferably 1024x1024",
                        "format": "JPEG or PNG",
                        "color_space": "sRGB",
                        "compression": "Minimal compression to preserve detail"
                    },
                    
                    "photography_standards": {
                        "lighting": [
                            "Natural daylight preferred",
                            "Avoid harsh shadows",
                            "Consistent illumination",
                            "No flash unless diffused"
                        ],
                        "composition": [
                            "Face fills 70-80% of frame",
                            "Direct frontal view",
                            "Clean background",
                            "No makeup or minimal makeup"
                        ],
                        "distance": "30-50cm from subject",
                        "focus": "Sharp focus on skin texture"
                    }
                },
                
                "quality_control": {
                    "image_validation": [
                        "Check for proper exposure",
                        "Verify skin visibility and detail",
                        "Ensure consistent labeling",
                        "Remove poor quality images"
                    ],
                    
                    "labeling_validation": [
                        "Multiple expert reviews",
                        "Inter-annotator agreement",
                        "Difficult cases review",
                        "Regular calibration sessions"
                    ],
                    
                    "bias_prevention": [
                        "Diverse demographic representation",
                        "Multiple lighting conditions",
                        "Various skin tones and ethnicities",
                        "Age range representation"
                    ]
                },
                
                "data_organization": {
                    "directory_structure": {
                        "dataset/": {
                            "train/": {
                                "normal/": "Training images for normal skin",
                                "dry/": "Training images for dry skin",
                                "oily/": "Training images for oily skin",
                                "combination/": "Training images for combination skin",
                                "sensitive/": "Training images for sensitive skin"
                            },
                            "validation/": "Same structure as train/",
                            "test/": "Same structure as train/",
                            "metadata/": {
                                "annotations.json": "Detailed annotations",
                                "demographics.json": "Subject demographics",
                                "quality_scores.json": "Image quality metrics"
                            }
                        }
                    },
                    
                    "metadata_requirements": [
                        "Subject age range",
                        "Skin tone (Fitzpatrick scale)",
                        "Gender",
                        "Lighting conditions",
                        "Camera/device used",
                        "Date and time",
                        "Annotator ID",
                        "Confidence score"
                    ]
                }
            }
        }
        
        return guide
    
    def create_annotation_interface(self):
        """Create a simple annotation interface for manual labeling"""
        
        interface_code = '''
<!DOCTYPE html>
<html>
<head>
    <title>Skin Type Annotation Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .image-container { display: flex; gap: 20px; margin-bottom: 20px; }
        .image-display { flex: 1; }
        .annotation-panel { flex: 1; background: #f5f5f5; padding: 20px; border-radius: 8px; }
        img { max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 8px; }
        .skin-type-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 20px 0; }
        button { padding: 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .skin-type { background: #e0e0e0; }
        .skin-type.selected { background: #4CAF50; color: white; }
        .confidence-slider { margin: 20px 0; }
        .notes { width: 100%; height: 80px; margin: 10px 0; padding: 8px; border: 1px solid #ddd; }
        .submit-btn { background: #2196F3; color: white; padding: 15px 30px; font-size: 18px; }
    </style>
</head>
<body>
    <h1>Skin Type Annotation Interface</h1>
    
    <div class="image-container">
        <div class="image-display">
            <img id="currentImage" src="" alt="Skin image to annotate">
            <div>
                <button onclick="previousImage()">← Previous</button>
                <span id="imageCounter">1 / 100</span>
                <button onclick="nextImage()">Next →</button>
            </div>
        </div>
        
        <div class="annotation-panel">
            <h3>Select Skin Type:</h3>
            <div class="skin-type-buttons">
                <button class="skin-type" onclick="selectSkinType('normal')">Normal</button>
                <button class="skin-type" onclick="selectSkinType('dry')">Dry</button>
                <button class="skin-type" onclick="selectSkinType('oily')">Oily</button>
                <button class="skin-type" onclick="selectSkinType('combination')">Combination</button>
                <button class="skin-type" onclick="selectSkinType('sensitive')">Sensitive</button>
            </div>
            
            <h3>Confidence Level:</h3>
            <div class="confidence-slider">
                <input type="range" id="confidence" min="1" max="10" value="5">
                <span id="confidenceValue">5/10</span>
            </div>
            
            <h3>Notes:</h3>
            <textarea class="notes" id="notes" placeholder="Additional observations about the skin..."></textarea>
            
            <h3>Visible Characteristics:</h3>
            <label><input type="checkbox" id="shininess"> Shininess/Oil</label><br>
            <label><input type="checkbox" id="dryness"> Dryness/Flaking</label><br>
            <label><input type="checkbox" id="redness"> Redness/Irritation</label><br>
            <label><input type="checkbox" id="pores"> Visible Pores</label><br>
            <label><input type="checkbox" id="texture"> Uneven Texture</label><br>
            
            <button class="submit-btn" onclick="saveAnnotation()">Save Annotation</button>
        </div>
    </div>
    
    <script>
        let currentImageIndex = 0;
        let selectedSkinType = '';
        let annotations = [];
        
        function selectSkinType(type) {
            document.querySelectorAll('.skin-type').forEach(btn => btn.classList.remove('selected'));
            event.target.classList.add('selected');
            selectedSkinType = type;
        }
        
        function saveAnnotation() {
            const annotation = {
                image: currentImageIndex,
                skinType: selectedSkinType,
                confidence: document.getElementById('confidence').value,
                notes: document.getElementById('notes').value,
                characteristics: {
                    shininess: document.getElementById('shininess').checked,
                    dryness: document.getElementById('dryness').checked,
                    redness: document.getElementById('redness').checked,
                    pores: document.getElementById('pores').checked,
                    texture: document.getElementById('texture').checked
                },
                timestamp: new Date().toISOString()
            };
            
            annotations.push(annotation);
            console.log('Annotation saved:', annotation);
            
            // Move to next image
            nextImage();
        }
        
        // Add more JavaScript functions for image navigation
    </script>
</body>
</html>
        '''
        
        return interface_code
    
    def analyze_dataset_quality(self, dataset_path):
        """Analyze the quality of an existing dataset"""
        
        if not os.path.exists(dataset_path):
            return {"error": "Dataset path not found"}
        
        analysis = {
            "dataset_statistics": defaultdict(int),
            "quality_metrics": {},
            "recommendations": []
        }
        
        # Count images per category
        for skin_type in self.skin_types.keys():
            type_path = os.path.join(dataset_path, skin_type)
            if os.path.exists(type_path):
                images = [f for f in os.listdir(type_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                analysis["dataset_statistics"][skin_type] = len(images)
                
                # Quality analysis for first few images
                if images:
                    self._analyze_image_quality(
                        os.path.join(type_path, images[0]), 
                        analysis["quality_metrics"]
                    )
        
        # Generate recommendations
        total_images = sum(analysis["dataset_statistics"].values())
        
        if total_images < 1000:
            analysis["recommendations"].append("Consider collecting more images (minimum 1000 total)")
        
        # Check for imbalance
        counts = list(analysis["dataset_statistics"].values())
        if counts and max(counts) / max(min(counts), 1) > 2:
            analysis["recommendations"].append("Dataset is imbalanced - collect more images for underrepresented classes")
        
        return analysis
    
    def _analyze_image_quality(self, image_path, quality_metrics):
        """Analyze individual image quality"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return
            
            # Check resolution
            height, width = img.shape[:2]
            quality_metrics["average_resolution"] = f"{width}x{height}"
            
            # Check brightness
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            quality_metrics["average_brightness"] = brightness
            
            # Check sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics["sharpness_score"] = laplacian_var
            
        except Exception as e:
            quality_metrics["analysis_error"] = str(e)

def main():
    """Main function to generate data collection guide"""
    
    improver = SkinDatasetImprover()
    
    # Generate comprehensive guide
    guide = improver.create_data_collection_guide()
    
    # Save guide as JSON
    with open("data_collection_guide.json", 'w') as f:
        json.dump(guide, f, indent=2)
    
    # Create annotation interface
    interface_html = improver.create_annotation_interface()
    
    with open("annotation_interface.html", 'w') as f:
        f.write(interface_html)
    
    print("📚 DATA COLLECTION GUIDE GENERATED")
    print("="*45)
    print("✅ Saved comprehensive guide to: data_collection_guide.json")
    print("✅ Saved annotation interface to: annotation_interface.html")
    print()
    print("🎯 KEY RECOMMENDATIONS FOR ACCURACY:")
    print("-" * 40)
    print("1. 📸 Collect 500+ images per skin type")
    print("2. 🌈 Include diverse demographics and lighting")
    print("3. 👨‍⚕️ Get expert dermatologist validation")
    print("4. 🔍 Use high-resolution, sharp images")
    print("5. 📊 Maintain balanced dataset")
    print("6. ✋ Implement quality control checks")
    print("7. 🔄 Continuously validate and improve")

if __name__ == "__main__":
    main()
