import './SummaryPanel.css';

function SummaryPanel() {
    const segments = [
        { id: 1, time: '00:00 - 01:30', title: '인트로 및 주제 소개', summary: '강의자가 오늘의 주제인 "머신러닝 기초"에 대해 소개하고, 강의 목표와 진행 순서를 설명합니다.', status: 'done' },
        { id: 2, time: '01:30 - 04:15', title: '지도학습의 개념', summary: '레이블이 있는 데이터를 사용하여 모델을 학습시키는 지도학습의 기본 원리를 설명합니다. 분류와 회귀 문제의 차이점도 다룹니다.', status: 'done' },
        { id: 3, time: '04:15 - 06:00', title: '비지도학습 소개', summary: '레이블 없이 데이터의 패턴을 찾는 비지도학습에 대해 설명합니다. 클러스터링과 차원 축소의 예시를 제시합니다.', status: 'progress' },
        { id: 4, time: '06:00 - 07:15', title: '실제 적용 사례', summary: '', status: 'pending' },
    ];

    return (
        <div className="summary-panel">
            <div className="summary-header">
                <h3>📝 실시간 요약</h3>
                <div className="progress-info">
                    <div className="mini-progress">
                        <div className="mini-progress-fill" style={{ width: '65%' }}></div>
                    </div>
                    <span>3/4 세그먼트 완료</span>
                </div>
            </div>

            <div className="summary-timeline">
                {segments.map((segment) => (
                    <div key={segment.id} className={`segment ${segment.status}`}>
                        <div className="segment-marker">
                            {segment.status === 'done' && '✓'}
                            {segment.status === 'progress' && <span className="loading-dot"></span>}
                            {segment.status === 'pending' && ''}
                        </div>
                        <div className="segment-content">
                            <div className="segment-header">
                                <span className="segment-time">{segment.time}</span>
                                <span className="segment-title">{segment.title}</span>
                            </div>
                            {segment.summary ? (
                                <p className="segment-summary">{segment.summary}</p>
                            ) : (
                                <p className="segment-pending">분석 대기 중...</p>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default SummaryPanel;
