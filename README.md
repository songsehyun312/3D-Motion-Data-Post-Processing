## 파일 설명
| 파일 | 설명 |
|---|---|
| `p1_create_raw_keypoints.py` | 영상에서 3D 키포인트 추출 |
| `p1_2_post_processing.py` | 바닥 보정 후처리 |
| `p5_analysis.py` | 후처리 전후 비교 분석, smoothing, 그래프 생성 |
| `p6_visualize.py` | raw / 제공 보정 / smoothing 결과를 3D로 나란히 보여주는 영상 생성 |
| `dance_compare.mp4` | `p6_visualize.py` 결과 영상 |

## 실행 방법
```
python scripts/p5_analysis.py dance
python scripts/p6_visualize.py dance
```
