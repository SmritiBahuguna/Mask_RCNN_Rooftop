# Rooftop Detection using Mask RCNN

To detect rooftop from satellite images for Indian rooftops, using Mask RCNN.

## Getting Started

mrcnn contains the model configuration a few of which are overridden according to our requirements in custom.py and evaluation using mAP metric in eval.py. Test the trained model for test dataset using test_code.py.

rooftop_dataset contains:
* train - Training Images with annotation file
* val - Validation Images with annotation file
* test - Test Images

## Training

python3 custom.py train --dataset=solar_panel_dataset --weights=coco

## Evaluation:

python3 eval.py 

## Testing:

python3 test_code.py 

## Results:
After being trained on 102 images, 
* mAP : 0.86

## References

* Paper Link: [Mask R-CNN](https://arxiv.org/abs/1703.06870).
Link to the repository: https://github.com/matterport/Mask_RCNN

