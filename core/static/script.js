var videoContainer = document.getElementById('videoContainer');
var video = document.getElementById('videoPlayer');
var youtubeVideoContainer = document.getElementById('youtubeVideoContainer');
var youtubeVideo = youtubeVideoContainer.querySelector('iframe');
var startTime = 0;
var endTime = 0;


function playVideo(videoPath, start, end, frameRate) {
    if (videoPath.includes('youtube.com')) {
        // Thêm tham số start vào URL của video
        var videoId = videoPath.match(/[?&]v=([^&]+)/)[1];
        var embedURL = `https://www.youtube.com/embed/${videoId}?autoplay=1&start=${start}`;
        youtubeVideo.src = embedURL;
        youtubeVideoContainer.style.display = 'block';    
    }
    else {
        // Nếu videoPath là đường dẫn trong thư mục, sử dụng thẻ video
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
    // hien thi frame index
    const frameIndexElement = document.getElementById('frameIndex');

    video.addEventListener('timeupdate', () => {
      // Lấy thời gian hiện tại của video
      const currentTime = video.currentTime;

      // Tính toán frame index tương ứng
      const frameIndex = Math.floor(currentTime * frameRate);

      // Cập nhật giá trị frame index trên trang web
      frameIndexElement.textContent = frameIndex;
    });
}


videoContainer.addEventListener('click', function (event) {
    if (event.target === videoContainer) {
        video.pause();
        videoContainer.style.display = 'none';
    }
});
youtubeVideoContainer.addEventListener('click', function (event) {
    if (event.target === youtubeVideoContainer) {
        youtubeVideo.src = '';
        youtubeVideoContainer.style.display = 'none';
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

