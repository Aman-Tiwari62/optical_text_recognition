from pathlib import Path

from PIL import Image

from src.dataset import parse_iam_annotation_line, validate_iam_annotations


def test_validate_iam_annotations(tmp_path):
    image_root = tmp_path / "data" / "raw" / "iam_words" / "words"
    image_dir = image_root / "a01" / "a01-000u"
    image_dir.mkdir(parents=True)

    image_path = image_dir / "a01-000u-00-00.png"
    image_path_2 = image_dir / "a01-000u-00-01.png"
    image = Image.new("L", (64, 32), color=255)
    image.save(image_path)
    image.save(image_path_2)

    annotation_file = tmp_path / "data" / "raw" / "iam_words" / "words.txt"
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    annotation_file.write_text(
        "# comment\n"
        "a01-000u-00-00 ok 154 24 10 25 12 AT A\n"
        "a01-000u-00-01 ok 154 30 10 20 14 NN MOVE\n",
        encoding="utf-8",
    )

    stats = validate_iam_annotations(annotation_file, image_root, include_err=False)

    assert stats["total_annotation_records"] == 2
    assert stats["successfully_loaded_image_label_pairs"] == 2
    assert stats["missing_images"] == 0
    assert stats["malformed_records"] == 0
    assert stats["duplicate_ids"] == 0
    assert stats["sample_records"][0]["transcription"] == "A"
    assert stats["status_counts"] == {"ok": 2, "err": 0}


def test_iam_parser_uses_actual_fields_and_preserves_transcription_spaces():
    record, reason = parse_iam_annotation_line(
        "a01-000u-00-00 ok 154 408 768 27 51 AT TWO WORDS"
    )

    assert reason is None
    assert record is not None
    assert (record.graylevel, record.x, record.y, record.width, record.height) == (
        154,
        408,
        768,
        27,
        51,
    )
    assert record.tag == "AT"
    assert record.transcription == "TWO WORDS"
