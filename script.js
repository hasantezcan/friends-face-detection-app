const imageUpload = document.getElementById('imageUpload')

// for local fetch
//const localhost = 'http://127.0.0.1:5500/' 

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)

async function start() {
    const container = document.createElement('div')
    container.classList.add("result-image")
    container.style.position = 'relative'
    document.body.append(container)
    const labeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
    
    let image
    let canvas

    document.body.append('All models are loaded')

    imageUpload.addEventListener('change', async () => {
        if (image) image.remove()
        if (canvas) canvas.remove()
        image = await faceapi.bufferToImage(imageUpload.files[0])
        container.append(image)
        canvas = faceapi.createCanvasFromMedia(image)
        container.append(canvas)
        const displaySize = { width: image.width, height: image.height }
        faceapi.matchDimensions(canvas, displaySize)

        const detections = await faceapi
        .detectAllFaces(image)
        .withFaceLandmarks()
        .withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, displaySize)

        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
        results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
            drawBox.draw(canvas)
        })
    })
}

function loadLabeledImages() {
    const labels = [
        'Chandler Bing',
        'Ross Geller',
        'Joey Tribbiani',
        'Monica Geller',
        'Rachel Green',
        'Phoebe Buffay',
        // 'Gunther',
        // 'Janice'
    ]

    return Promise.all(
        labels.map(async label => {
            const descriptions = []
            for (let i = 1; i <= 3; i++) {
                // local fetch
                // const img = await faceapi.fetchImage(`${localhost}/img/faces/${label}/${i}.jpg`)

                // github fetch
                const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/hasantezcan/friends-face-detection-app/master/img/faces/${label}/${i}.jpg`)


                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }

            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}
