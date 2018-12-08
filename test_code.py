# Load a random image from the images folder
for file_names in os.listdir(IMAGE_DIR):
  #file_names = next(os.walk(IMAGE_DIR))[2]
  #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
  image = skimage.io.imread(os.path.join(IMAGE_DIR,file_names))

  # Run detection
  results = model.detect([image], verbose=1)

  # Visualize results
  r = results[0]
  visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                              dataset.class_names, r['scores'], show_bbox=False)