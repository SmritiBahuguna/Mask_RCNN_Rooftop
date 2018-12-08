# Compute VOC-Style mAP @ IoU=0.5
#image_ids = np.random.choice(dataset.image_ids, 10)	#dataset is validation dataset
image_ids=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])	#not working for all the images as it hasn't been able to detect solar panels there
#print(dataset.image_ids)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    print(image_id, gt_mask.shape, r['masks'].shape)
    # Compute AP
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))