let previousSleep = false;
let previousDeepSleep = false;
let previousYawn = false;

let soundEnabled = false;


// --------------------------------------------------
// Meme audio
// --------------------------------------------------

const sleepAudio = new Audio(
    "/static/audio/sleep.mp3"
);

const deepSleepAudio = new Audio(
    "/static/audio/deep_sleep.mp3"
);

const yawnAudio = new Audio(
    "/static/audio/yawn.mp3"
);


function playSound(audio) {

    if (!soundEnabled) {
        return;
    }

    audio.currentTime = 0;

    audio.play().catch(error => {
        console.log("Audio playback blocked:", error);
    });
}


// --------------------------------------------------
// Start button
// --------------------------------------------------

document
    .getElementById("start-button")
    .addEventListener("click", function () {

        soundEnabled = true;

        this.textContent = "✓ DETECTION RUNNING";

        this.classList.add("started");

        document
            .getElementById("start-message")
            .textContent =
            "Meme sounds enabled 🔊";

        // Unlock audio
        sleepAudio.load();
        deepSleepAudio.load();
        yawnAudio.load();

    });


// --------------------------------------------------
// Status update
// --------------------------------------------------

async function updateStatus() {

    try {

        const response = await fetch("/status");

        const data = await response.json();


        // Update cards

        document.getElementById(
            "face-status"
        ).textContent = data.face;

        document.getElementById(
            "eyes-status"
        ).textContent = data.eyes;

        document.getElementById(
            "head-status"
        ).textContent = data.head;

        document.getElementById(
            "body-status"
        ).textContent = data.body;

        document.getElementById(
            "yawn-status"
        ).textContent = data.yawn;

        document.getElementById(
            "overall-status"
        ).textContent = data.overall;


        // --------------------------------------------------
        // Detection states
        // --------------------------------------------------

        const isSleeping =
            data.face === "LIKELY SLEEPING";

        const isDeepSleeping =
            data.body === "LIKELY DEEP SLEEPING";

        const isYawning =
            data.yawn === "YAWNING - SLEEPY";


        // --------------------------------------------------
        // Play only when state begins
        // --------------------------------------------------

        if (isSleeping && !previousSleep) {

            playSound(sleepAudio);
        }


        if (isDeepSleeping && !previousDeepSleep) {

            playSound(deepSleepAudio);
        }


        if (isYawning && !previousYawn) {

            playSound(yawnAudio);
        }


        // Remember states

        previousSleep = isSleeping;
        previousDeepSleep = isDeepSleeping;
        previousYawn = isYawning;


    } catch (error) {

        console.log(
            "Could not get detection status"
        );
    }
}


// Update every 200 ms

setInterval(updateStatus, 200);

updateStatus();