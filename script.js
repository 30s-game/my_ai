let model;
let vocab = [];
let responses = [];

// 1. JSON 데이터 로드 및 초기화
async function initAI() {
    const status = document.getElementById('status');
    status.innerText = "데이터 로딩 중...";

    try {
        const response = await fetch('data.json');
        const data = await response.json();
        
        vocab = data.training_data.map(d => d.question);
        responses = data.training_data.map(d => d.answer);

        await train(status);
    } catch (e) {
        status.innerText = "오류: data.json 파일을 찾을 수 없습니다.";
        console.error(e);
    }
}

// 2. 신경망 학습 (Deep Learning)
async function train(statusTag) {
    statusTag.innerText = "신경망 학습 중... 잠시만 기다려주세요.";

    // 모델 구조 정의
    model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [vocab.length], units: 16, activation: 'relu'}));
    model.add(tf.layers.dense({units: 8, activation: 'relu'}));
    model.add(tf.layers.dense({units: responses.length, activation: 'softmax'}));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // 데이터를 텐서(Tensor) 형태로 변환
    const xs = tf.oneHot(tf.tensor1d(vocab.map((_, i) => i), 'int32'), vocab.length);
    const ys = tf.oneHot(tf.tensor1d(responses.map((_, i) => i), 'int32'), responses.length);

    // 학습 실행
    await model.fit(xs, ys, { epochs: 200 });

    statusTag.innerText = "학습 완료! 이제 대화가 가능합니다.";
    document.getElementById('userInput').disabled = false;
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('trainBtn').style.display = "none";
}

// 3. 메시지 전송 및 추론
async function sendMessage() {
    const input = document.getElementById('userInput');
    const text = input.value.trim();
    if (!text) return;

    appendMsg(text, 'user');
    input.value = '';

    // 입력 텍스트 벡터화
    const inputVec = new Array(vocab.length).fill(0);
    vocab.forEach((word, i) => { if (text.includes(word)) inputVec[i] = 1; });
    
    const prediction = model.predict(tf.tensor2d([inputVec]));
    const idx = prediction.argMax(-1).dataSync()[0];
    const confidence = prediction.max().dataSync()[0];

    setTimeout(() => {
        const reply = confidence > 0.5 ? responses[idx] : "무슨 말씀인지 잘 모르겠어요. 조금 더 쉽게 말해주실래요?";
        appendMsg(reply, 'ai');
    }, 500);
}

function appendMsg(text, sender) {
    const div = document.createElement('div');
    div.className = `msg ${sender}`;
    div.innerText = text;
    const box = document.getElementById('messages');
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}
