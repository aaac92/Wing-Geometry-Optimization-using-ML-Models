"""
Quick round-trip test for compression_Alma.

Creates a text file on the Desktop, compresses it into a .rar with
compress_folder(), decompresses that .rar with decompress_archive(),
and checks the extracted content matches the original.

Run from the same folder as file_handling.py (or adjust the import below).
"""
import shutil
import sys
from pathlib import Path
from file_handling_support import compression_Alma

# NOTE: if your Desktop folder is localized (e.g. "Escritorio" instead of
# "Desktop"), change this to that folder name.
DESKTOP = Path.home() / "Desktop"

SOURCE_DIR = DESKTOP / "compression_alma_test_source"
ARCHIVE_DIR = DESKTOP / "compression_alma_test_archive"
EXTRACT_DIR = DESKTOP / "compression_alma_test_extracted"

TEST_FILENAME = "test_file.txt"
TEST_CONTENT = "compression_Alma round-trip test\nsome sample content 12345"
ARCHIVE_NAME = "test_archive"  # .rar gets appended automatically


def cleanup():
    for d in (SOURCE_DIR, ARCHIVE_DIR, EXTRACT_DIR):
        shutil.rmtree(d, ignore_errors=True)


def main():
    cleanup()  # start from a clean state in case a previous run left files behind

    try:
        # 1. Create the source file to compress
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        test_file = SOURCE_DIR / TEST_FILENAME
        test_file.write_text(TEST_CONTENT)

        # 2. Compress
        rar_Alma = compression_Alma()
        rar_Alma.compress_folder(SOURCE_DIR, ARCHIVE_DIR, ARCHIVE_NAME, verbose=True)

        archive_path = ARCHIVE_DIR / f"{ARCHIVE_NAME}.rar"
        if not archive_path.exists():
            print(f"FAIL: archive was not created at {archive_path}")
            sys.exit(1)

        # 3. Decompress
        rar_Alma.decompress_archive(archive_path, EXTRACT_DIR, verbose=True)

        extracted_file = EXTRACT_DIR / TEST_FILENAME
        if not extracted_file.exists():
            print(f"FAIL: extracted file not found at {extracted_file}")
            sys.exit(1)

        # 4. Verify content round-tripped correctly
        extracted_content = extracted_file.read_text()
        if extracted_content != TEST_CONTENT:
            print("FAIL: extracted content does not match original")
            print(f"  expected: {TEST_CONTENT!r}")
            print(f"  got:      {extracted_content!r}")
            sys.exit(1)

        print("PASS: compress + decompress round-trip successful")

    finally:
        cleanup()


if __name__ == "__main__":
    main()