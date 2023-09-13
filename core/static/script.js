var videoContainer = document.getElementById('videoContainer');
var video = document.getElementById('videoPlayer');
var startTime = 0;
var endTime = 0;

function playVideo(videoPath, start, end) {
    video.src = videoPath;
    startTime = start;
    endTime = end;
    video.currentTime = startTime;
    videoContainer.style.display = 'block';

    var intervalId = setInterval(function () {
        if (video.currentTime >= endTime) {
            video.pause();
            videoContainer.style.display = 'none';
            clearInterval(intervalId);
        }
    }, 1000);

    video.play();
}

videoContainer.addEventListener('click', function (event) {
    if (event.target === videoContainer) {
        video.pause();
        videoContainer.style.display = 'none';
    }
});

function redirectToImages(imageUrl) {
    // Chuyển hướng trang web đến '/images'
    window.location.href = "/images?image_url=" + imageUrl;
}