try:
    from docx2markdown._docx_to_markdown import docx_to_markdown
except ImportError:
    docx_to_markdown = None
import openai
import requests
import mimetypes
import json
import re
import shutil
import tempfile
from retry import retry
from typing import List, Dict, Any, Optional, Tuple, Literal
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urlparse, urljoin
import os
import subprocess
try:
    import xmltodict
except ImportError:
    xmltodict = None
import asyncio
import nest_asyncio
from app.cosight.tool.excel_toolkit import extract_excel_content
from app.common.logger_util import logger

nest_asyncio.apply()


class DocumentProcessingToolkit:
    r"""A class representing a toolkit for processing document and return the content of the document.

    This class provides methods for extracting text from documents including
    legacy Word .doc, .docx, PDF, Excel, CSV, text, JSON, XML, ZIP, and webpages.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = "tmp/"
        if cache_dir:
            self.cache_dir = cache_dir

        proxy = os.environ.get("PROXY")
        self.proxies = {"http": proxy, "https": proxy} if proxy else None

    @retry((requests.RequestException))
    def extract_document_content(self, document_path: str) -> Tuple[bool, str]:
        r"""Extract the content of a given document (or url) and return the processed text.
        It may filter out some information, resulting in inaccurate content.

        Args:
            document_path (str): The path of the document to be processed, either a local path or a URL. It can process legacy Word .doc, .docx, PDF, Excel, CSV, text, JSON, XML, ZIP, and webpages.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the document was processed successfully, and the content of the document (if success).
        """
        logger.info(f"Calling extract_document_content function with document_path=`{document_path}`")
        lower_document_path = document_path.lower()

        if any(lower_document_path.endswith(ext) for ext in ['txt', 'html', 'md']):
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            f.close()
            return content

        if any(lower_document_path.endswith(ext) for ext in ['zip']):
            extracted_files = self._unzip_file(document_path)
            return f"The extracted files are: {extracted_files}"

        if any(lower_document_path.endswith(ext) for ext in ['json', 'jsonl', 'jsonld']):
            with open(document_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            f.close()
            return content

        if any(lower_document_path.endswith(ext) for ext in ['py']):
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            f.close()
            return content

        if any(lower_document_path.endswith(ext) for ext in ['xlsx', 'xls', 'csv']):
            content = extract_excel_content(document_path)
            return content

        if any(lower_document_path.endswith(ext) for ext in ['xml']):
            data = None
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            f.close()

            if xmltodict is None:
                return content

            try:
                data = xmltodict.parse(content)
                logger.info(f"The extracted xml data is: {data}")
                return data

            except Exception as e:
                logger.error(f"raise error: {str(e)}, The raw xml data is: {content}", exc_info=True)
                return content

        if self._is_webpage(document_path):
            extracted_text = self._extract_webpage_content(document_path)
            return extracted_text


        else:
            # judge if url
            parsed_url = urlparse(document_path)
            is_url = all([parsed_url.scheme, parsed_url.netloc])
            if not is_url:
                if not os.path.exists(document_path):
                    return f"Document not found at path: {document_path}."

            # if is docx file, use docx2markdown to convert it
            if lower_document_path.endswith(".docx"):
                if docx_to_markdown is None:
                    return (
                        "Failed to extract .docx content because docx2markdown "
                        "is not installed in the runtime environment."
                    )
                if is_url:
                    tmp_path = self._download_file(document_path)
                else:
                    tmp_path = document_path

                file_name = os.path.basename(tmp_path)
                md_file_path = f"{file_name}.md"
                docx_to_markdown(tmp_path, md_file_path)

                # load content of md file
                with open(md_file_path, "r") as f:
                    extracted_text = f.read()
                f.close()
                return extracted_text
            if lower_document_path.endswith(".doc"):
                if is_url:
                    tmp_path = self._download_file(document_path)
                    if not tmp_path:
                        return f"Failed to download document: {document_path}"
                else:
                    tmp_path = document_path
                return self._extract_legacy_doc_content(tmp_path)
            if lower_document_path.endswith(".pdf"):
                if is_url:
                    tmp_path = self._download_file(document_path)
                    if not tmp_path:
                        return f"Failed to download document: {document_path}"
                    document_path = tmp_path
                return self._extract_pdf_content(document_path)
            return ""

    def _extract_pdf_content(self, document_path: str) -> str:
        extractors = (
            ("pypdf", self._extract_pdf_with_pypdf),
            ("pdfplumber", self._extract_pdf_with_pdfplumber),
            ("PyMuPDF", self._extract_pdf_with_pymupdf),
        )
        errors = []
        for name, extractor in extractors:
            try:
                extracted_text = self._clean_extracted_text(extractor(document_path))
            except ImportError as exc:
                errors.append(f"{name}: {exc}")
                continue
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                logger.warning(f"PDF extractor {name} failed for {document_path}: {exc}")
                continue

            if extracted_text:
                return extracted_text

        return (
            "Failed to extract text from PDF file. The file may be scanned, "
            "image-only, encrypted, or unsupported by installed parsers. "
            f"Parser errors: {'; '.join(errors)}"
        )

    def _extract_pdf_with_pypdf(self, document_path: str) -> str:
        from pypdf import PdfReader

        with open(document_path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _extract_pdf_with_pdfplumber(self, document_path: str) -> str:
        import pdfplumber

        with pdfplumber.open(document_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    def _extract_pdf_with_pymupdf(self, document_path: str) -> str:
        import fitz

        with fitz.open(document_path) as document:
            return "\n".join(page.get_text("text") or "" for page in document)

    def _extract_legacy_doc_content(self, document_path: str) -> str:
        """Extract text from old binary Word .doc files.

        Prefer installed converters when available, then fall back to a local
        Unicode-string extraction pass for environments without office tools.
        """
        command_text = self._extract_legacy_doc_with_commands(document_path)
        if command_text:
            return command_text

        fallback_text = self._extract_legacy_doc_with_unicode_strings(document_path)
        if fallback_text:
            return (
                "Text extracted from legacy Word .doc using best-effort Unicode "
                "string recovery. Formatting, tables, and page numbers may be incomplete.\n\n"
                + fallback_text
            )

        return (
            "Failed to extract text from legacy Word .doc file. Install one of "
            "LibreOffice/soffice, antiword, catdoc, or wvText in the runtime "
            f"environment, then retry. Document path: {document_path}"
        )

    def _extract_legacy_doc_with_commands(self, document_path: str) -> str:
        office_text = self._extract_legacy_doc_with_office(document_path)
        if self._has_meaningful_text(office_text):
            return office_text

        wv_text = self._extract_legacy_doc_with_wvtext(document_path)
        if self._has_meaningful_text(wv_text):
            return wv_text

        for command in (
            ["antiword", "-m", "UTF-8.txt", document_path],
            ["antiword", document_path],
            ["catdoc", "-w", document_path],
        ):
            if not shutil.which(command[0]):
                continue
            text = self._run_text_extraction_command(command)
            if self._has_meaningful_text(text):
                return text

        return ""

    def _extract_legacy_doc_with_wvtext(self, document_path: str) -> str:
        if not shutil.which("wvText"):
            return ""

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(
                tmp_dir,
                os.path.splitext(os.path.basename(document_path))[0] + ".txt",
            )
            result = subprocess.run(
                ["wvText", document_path, output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._doc_extract_timeout(),
                check=False,
            )
            if result.returncode != 0:
                logger.warning(
                    f"wvText failed for {document_path}: "
                    f"{self._decode_text_bytes(result.stderr).strip()}"
                )
                return ""
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    return self._clean_extracted_text(self._decode_text_bytes(f.read()))
        return ""

    def _extract_legacy_doc_with_office(self, document_path: str) -> str:
        office_bin = next(
            (binary for binary in ("soffice", "libreoffice") if shutil.which(binary)),
            None,
        )
        if not office_bin:
            return ""

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [
                    office_bin,
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    tmp_dir,
                    document_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._doc_extract_timeout(),
                check=False,
            )
            if result.returncode != 0:
                logger.warning(
                    f"{office_bin} conversion failed for {document_path}: "
                    f"{self._decode_text_bytes(result.stderr).strip()}"
                )
                return ""

            for name in os.listdir(tmp_dir):
                if name.lower().endswith(".txt"):
                    with open(os.path.join(tmp_dir, name), "rb") as f:
                        return self._clean_extracted_text(self._decode_text_bytes(f.read()))
        return ""

    def _run_text_extraction_command(self, command: List[str]) -> str:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._doc_extract_timeout(),
                check=False,
            )
        except Exception as exc:
            logger.warning(f"Legacy .doc extractor failed to run {command[0]}: {exc}")
            return ""

        if result.returncode != 0:
            logger.warning(
                f"Legacy .doc extractor {command[0]} failed: "
                f"{self._decode_text_bytes(result.stderr).strip()}"
            )
            return ""

        return self._clean_extracted_text(self._decode_text_bytes(result.stdout))

    def _extract_legacy_doc_with_unicode_strings(self, document_path: str) -> str:
        try:
            with open(document_path, "rb") as f:
                data = f.read()
        except Exception as exc:
            logger.error(f"Failed to read legacy .doc file {document_path}: {exc}", exc_info=True)
            return ""

        candidates = []
        for encoding in ("utf-16le", "utf-16be"):
            decoded = data.decode(encoding, errors="ignore")
            candidates.append(self._recover_text_chunks(decoded))

        text = max(candidates, key=self._text_score, default="")
        return text if self._has_meaningful_text(text) else ""

    def _recover_text_chunks(self, decoded_text: str) -> str:
        pattern = re.compile(
            r"[\u4e00-\u9fffA-Za-z0-9"
            r"\uff10-\uff19\uff21-\uff3a\uff41-\uff5a"
            r"\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65"
            r".,;:/%+\-=(){}\[\]_ \t\r\n]{8,}"
        )
        chunks = []
        seen = set()
        for chunk in pattern.findall(decoded_text):
            cleaned = self._clean_extracted_text(chunk)
            if not self._has_meaningful_text(cleaned, minimum_score=8):
                continue
            normalized = re.sub(r"\s+", "", cleaned)
            if normalized in seen:
                continue
            seen.add(normalized)
            chunks.append(cleaned)

        return "\n".join(chunks)

    def _clean_extracted_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\x00", "")
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
        text = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n\s+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _decode_text_bytes(self, data: bytes) -> str:
        for encoding in ("utf-8", "gb18030", "big5", "utf-16", "utf-16le", "latin1"):
            try:
                return data.decode(encoding)
            except UnicodeError:
                continue
        return data.decode("utf-8", errors="ignore")

    def _has_meaningful_text(self, text: str, minimum_score: int = 40) -> bool:
        return self._text_score(text) >= minimum_score

    def _text_score(self, text: str) -> int:
        if not text:
            return 0
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin_words = len(re.findall(r"[A-Za-z]{3,}", text))
        digits = len(re.findall(r"\d", text))
        return cjk * 2 + latin_words + min(digits, 100)

    def _doc_extract_timeout(self) -> int:
        raw = os.environ.get("DOC_EXTRACT_TIMEOUT", "120")
        try:
            return max(5, int(raw))
        except ValueError:
            return 120

    def _is_webpage(self, url: str) -> bool:
        r"""Judge whether the given URL is a webpage."""
        try:
            parsed_url = urlparse(url)
            is_url = all([parsed_url.scheme, parsed_url.netloc])
            if not is_url:
                return False

            path = parsed_url.path
            file_type, _ = mimetypes.guess_type(path)
            if 'text/html' in file_type:
                return True

            response = requests.head(url, allow_redirects=True, timeout=10, proxies=self.proxies)
            content_type = response.headers.get("Content-Type", "").lower()

            if "text/html" in content_type:
                return True
            else:
                return False

        except requests.exceptions.RequestException as e:
            # raise RuntimeError(f"Error while checking the URL: {e}")
            logger.error(f"Error while checking the URL: {str(e)}", exc_info=True)
            return False

        except TypeError:
            return True

    def _download_file(self, url: str):
        r"""Download a file from a URL and save it to the cache directory."""
        try:
            response = requests.get(url, stream=True, proxies=self.proxies)
            response.raise_for_status()
            file_name = url.split("/")[-1]

            file_path = os.path.join(self.cache_dir, file_name)

            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            return file_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading the file: {str(e)}", exc_info=True)

    def _get_formatted_time(self) -> str:
        import time
        return time.strftime("%m%d%H%M")

    def _unzip_file(self, zip_path: str) -> List[str]:
        if not zip_path.endswith('.zip'):
            raise ValueError("Only .zip files are supported")

        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_path = os.path.join(self.cache_dir, zip_name)
        os.makedirs(extract_path, exist_ok=True)

        try:
            subprocess.run(["unzip", "-o", zip_path, "-d", extract_path], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to unzip file: {e}")

        extracted_files = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                extracted_files.append(os.path.join(root, file))

        return extracted_files
