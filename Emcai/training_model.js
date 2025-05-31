// training_model.js
// Melatih CNN untuk mengenali gambar (doodle) menggunakan TensorFlow.js

class CNN {
	constructor(numClasses) {
		this.NUM_CLASSES = numClasses;
		this.IMAGE_SIZE = 784; // 28x28
		this.NUM_TRAIN_IMAGES = 400;
		this.NUM_TEST_IMAGES = 100;
		this.TRAIN_ITERATIONS = 50;
		this.TRAIN_BATCH_SIZE = 100;
		this.TEST_FREQUENCY = 5;
		this.TEST_BATCH_SIZE = 50;

		this.trainIteration = 0;
		this.aLoss = [];
		this.aAccu = [];

		const totalTrain = this.NUM_CLASSES * this.NUM_TRAIN_IMAGES;
		const totalTest = this.NUM_CLASSES * this.NUM_TEST_IMAGES;

		this.aTrainImages = new Float32Array(totalTrain * this.IMAGE_SIZE);
		this.aTrainClasses = new Uint8Array(totalTrain);
		this.aTrainIndices = tf.util.createShuffledIndices(totalTrain);
		this.trainElement = -1;

		this.aTestImages = new Float32Array(totalTest * this.IMAGE_SIZE);
		this.aTestClasses = new Uint8Array(totalTest);
		this.aTestIndices = tf.util.createShuffledIndices(totalTest);
		this.testElement = -1;

		this.model = tf.sequential();
		this.model.add(tf.layers.conv2d({
			inputShape: [28, 28, 1],
			kernelSize: 5,
			filters: 8,
			strides: 1,
			activation: 'relu',
			kernelInitializer: 'varianceScaling'
		}));
		this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
		this.model.add(tf.layers.conv2d({
			kernelSize: 5,
			filters: 16,
			strides: 1,
			activation: 'relu',
			kernelInitializer: 'varianceScaling'
		}));
		this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
		this.model.add(tf.layers.flatten());
		this.model.add(tf.layers.dense({
			units: this.NUM_CLASSES,
			kernelInitializer: 'varianceScaling',
			activation: 'softmax'
		}));
		this.model.compile({
			optimizer: tf.train.sgd(0.15),
			loss: 'categoricalCrossentropy',
			metrics: ['accuracy']
		});
	}

	splitDataset(imagesBuffer, datasetIndex) {
		const trainBuffer = new Float32Array(imagesBuffer.slice(0, this.IMAGE_SIZE * this.NUM_TRAIN_IMAGES))
			.map(val => val / 255.0);
		const testBuffer = new Float32Array(imagesBuffer.slice(this.IMAGE_SIZE * this.NUM_TRAIN_IMAGES))
			.map(val => val / 255.0);

		let startTrain = datasetIndex * this.NUM_TRAIN_IMAGES;
		this.aTrainImages.set(trainBuffer, startTrain * this.IMAGE_SIZE);
		this.aTrainClasses.fill(datasetIndex, startTrain, startTrain + this.NUM_TRAIN_IMAGES);

		let startTest = datasetIndex * this.NUM_TEST_IMAGES;
		this.aTestImages.set(testBuffer, startTest * this.IMAGE_SIZE);
		this.aTestClasses.fill(datasetIndex, startTest, startTest + this.NUM_TEST_IMAGES);
	}

	async train() {
		for (let i = 0; i < this.TRAIN_ITERATIONS; i++) {
			this.trainIteration++;

			const trainBatch = this.nextTrainBatch(this.TRAIN_BATCH_SIZE);
			let testBatch, validationSet;

			if (i % this.TEST_FREQUENCY === 0) {
				testBatch = this.nextTestBatch(this.TEST_BATCH_SIZE);
				validationSet = [testBatch.images, testBatch.labels];
			}

			const history = await this.model.fit(
				trainBatch.images,
				trainBatch.labels,
				{ batchSize: this.TRAIN_BATCH_SIZE, validationData: validationSet, epochs: 1 }
			);

			if (testBatch) {
				testBatch.images.dispose();
				testBatch.labels.dispose();
			}
			trainBatch.images.dispose();
			trainBatch.labels.dispose();

			await tf.nextFrame();
		}
	}

	nextTrainBatch(batchSize) {
		return this.fetchBatch(batchSize, this.aTrainImages, this.aTrainClasses, () => {
			this.trainElement = (this.trainElement + 1) % this.aTrainIndices.length;
			return this.aTrainIndices[this.trainElement];
		});
	}

	nextTestBatch(batchSize) {
		return this.fetchBatch(batchSize, this.aTestImages, this.aTestClasses, () => {
			this.testElement = (this.testElement + 1) % this.aTestIndices.length;
			return this.aTestIndices[this.testElement];
		});
	}

	fetchBatch(batchSize, aImages, aClasses, getIndex) {
		const batchImages = new Float32Array(batchSize * this.IMAGE_SIZE);
		const batchLabels = new Uint8Array(batchSize * this.NUM_CLASSES);

		for (let i = 0; i < batchSize; i++) {
			const idx = getIndex();
			const image = aImages.slice(idx * this.IMAGE_SIZE, (idx + 1) * this.IMAGE_SIZE);
			batchImages.set(image, i * this.IMAGE_SIZE);

			const label = new Uint8Array(this.NUM_CLASSES);
			label[aClasses[idx]] = 1;
			batchLabels.set(label, i * this.NUM_CLASSES);
		}

		const imagesTemp = tf.tensor2d(batchImages, [batchSize, this.IMAGE_SIZE]);
		const images = imagesTemp.reshape([batchSize, 28, 28, 1]);
		imagesTemp.dispose();

		const labels = tf.tensor2d(batchLabels, [batchSize, this.NUM_CLASSES]);
		return { images, labels };
	}
}

// Export if using in a module system
// module.exports = CNN;
