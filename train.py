import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Lambda
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


# Remap path
def dopath(source_path):
	filename = source_path.split('/')[-1]
	current_path = 'data/IMG/' + filename
	return current_path

# Open driving log
lines = []
with open( 'data/driving_log.csv' ) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# Create images and steering measurements arrays
images = []
measurements = []
for line in lines:
	
	# Get steering center
	steering_center = float(line[3])

	# Create adjusted steering measurements for the side camera images
	correction = 0.2
	steering_left = steering_center + correction
	steering_right = steering_center - correction

	# Read images
	image_center = cv2.imread( dopath( line[0] ) )
	image_left = cv2.imread( dopath( line[1] ) )
	image_right = cv2.imread( dopath( line[2] ) ) 

	# Add
	images.extend( [image_center, image_left, image_right] )
	measurements.extend( [steering_center, steering_left, steering_right] )

	# Flip images
	image_center_f = np.fliplr( image_center )
	image_left_f = np.fliplr( image_left )
	image_right_f = np.fliplr( image_right )

	# Add
	images.extend( [image_center_f, image_left_f, image_right_f] )
	measurements.extend( [-steering_center, -steering_left, -steering_right] )

# Training data
X_train = np.array( images )
y_train = np.array( measurements )


# Define model
model = Sequential()

# Crop
model.add( Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)) )

# Normalize
model.add( Lambda(lambda x: (x / 255.0) - 0.5) )
model.add( Convolution2D(6, 5, 5, activation="relu") )
model.add( MaxPooling2D() )
model.add( Convolution2D(6, 5, 5, activation="relu") )
model.add( MaxPooling2D() )
model.add( Flatten( ) )
model.add( Dense( 120 ) )
model.add( Dropout( 0.2 ) )
model.add( Dense( 84 ) )
model.add( Dense( 1 ) )

model.compile( loss='mse', optimizer='adam' )
model.fit( X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save( 'model.h5' )


