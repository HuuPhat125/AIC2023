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
    video.playbackRate = 2.0;
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



/// Tạo một khối mã JavaScript riêng biệt cho chức năng sắp xếp
function setupSortImages() {
    var imageContainers = document.querySelectorAll('.image-container');
    var imageContainersArray = Array.from(imageContainers);

    document.getElementById('sortImagesButton').addEventListener('click', function () {
        imageContainersArray.sort(function (a, b) {
            var videoNameA = a.getAttribute('data-video-name');
            var videoNameB = b.getAttribute('data-video-name');
            var frameIndexA = parseInt(a.getAttribute('data-frame-index'));
            var frameIndexB = parseInt(b.getAttribute('data-frame-index'));

            if (videoNameA === videoNameB) {
                return frameIndexA - frameIndexB;
            } else {
                return videoNameA.localeCompare(videoNameB);
            }
        });

        // Tạo một danh sách các đoạn hình ảnh dựa trên video name
        var videoSections = {};
        imageContainersArray.forEach(function (container) {
            var videoName = container.getAttribute('data-video-name');
            if (!videoSections[videoName]) {
                videoSections[videoName] = [];
            }
            videoSections[videoName].push(container);
        });

        // Xóa hình ảnh hiện tại
        imageContainersArray.forEach(function (container) {
            container.remove();
        });

        // Hiển thị hình ảnh theo từng đoạn và thêm tên video name
        for (var videoName in videoSections) {
            var section = document.createElement('div');
            section.className = 'video-section';
            section.textContent = videoName;
            document.body.appendChild(section);

            videoSections[videoName].forEach(function (container) {
                document.body.appendChild(container);
            });
        }
    });
}

// Gọi hàm setupSortImages để thiết lập chức năng sắp xếp
setupSortImages();

// Dưới đây có thể là các đoạn mã khác xử lý các sự kiện khác trong cùng tệp JavaScript
