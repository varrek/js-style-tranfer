document.getElementById('content-image-upload').addEventListener('change', handleContentImageUpload);
document.getElementById('style-image-upload').addEventListener('change', handleStyleImageUpload);
document.getElementById('apply-style-button').addEventListener('click', applyStyle);

let contentImageElement;
let styleImageElement;

function handleContentImageUpload(event) {
  const reader = new FileReader();
  reader.onload = function() {
    const imgElement = document.getElementById('content-image');
    imgElement.src = reader.result;
    contentImageElement = imgElement;
  };
  reader.readAsDataURL(event.target.files[0]);
}

function handleStyleImageUpload(event) {
  const reader = new FileReader();
  reader.onload = function() {
    const imgElement = document.getElementById('style-image');
    imgElement.src = reader.result;
    styleImageElement = imgElement;
  };
  reader.readAsDataURL(event.target.files[0]);
}

async function applyStyle() {
  if (contentImageElement && styleImageElement) {
    const styleModelUrl = 'https://raw.githubusercontent.com/reiinakano/arbitrary-image-stylization-tfjs/master/saved_model_style_js/model.json';
    const styleModel = await tf.loadGraphModel(styleModelUrl);

    const transformerModelUrl = 'https://raw.githubusercontent.com/reiinakano/arbitrary-image-stylization-tfjs/master/saved_model_transformer_js/model.json';
    const transformerModel = await tf.loadGraphModel(transformerModelUrl);

    const preprocess = (img) => {
      const tensor = tf.browser.fromPixels(img).toFloat().div(tf.scalar(255)).expandDims();
      return tf.image.resizeBilinear(tensor, [256, 256]);
    };

    const contentTensor = preprocess(contentImageElement);
    const styleTensor = preprocess(styleImageElement);

    const styleVector = await tf.tidy(() => {
      const style = styleTensor.cast('float32');
      return styleModel.execute({ 'Placeholder': style });
    });

    const reshapedStyleVector = styleVector.reshape([1, 1, 1, 100]);

    const stylizedImageTensor = await tf.tidy(() => {
      const content = contentTensor.cast('float32');

      const inputDict = {
        'Placeholder': content,
        'Placeholder_1': reshapedStyleVector,
      };

      return transformerModel.execute(inputDict);
    });

    const stylizedImageElement = document.getElementById('stylized-image');
    await tf.browser.toPixels(stylizedImageTensor.squeeze(), stylizedImageElement);
  } else {
    alert('Please upload both content and style images.');
  }
}
