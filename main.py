import os
import io
import uuid
import logging
import numpy as np
from uuid import uuid4
from typing import List
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse

from typing import List, Dict, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
import re
import pandas as pd
import fitz  # PyMuPDF

from fastapi.concurrency import run_in_threadpool
import re
from PIL import Image
from paddleocr import PaddleOCR
import paddle
from pydantic import BaseModel
from typing import List, Optional


class Item(BaseModel):
    item: Optional[str]
    gpn: Optional[str]
    description: Optional[str]
    country_of_origin: Optional[str]
    uom: Optional[str]
    quantity: Optional[str]
    fob_price_sgd: Optional[str]
    fob_amount_sgd: Optional[str]

class RecalculateRequest(BaseModel):
    items: List[Item]
    dividing_by: int


# Initialize FastAPI app
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PaddleOCR
ocr_engine = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    rec_batch_num=1,       # Set recognition batch size to 1
    # use_gpu=True,          # Explicitly enable GPU
    enable_mkldnn=False,   # Disable MKLDNN to avoid CPU fallback issues
    precision="fp32",      # Use FP32 (default) or try "fp16" for lower memory usage
    # det_max_side_len=960,  # Limit image size for detection to reduce memory
)

REGIONS = {
    "shipper": {'x1': 162, 'y1': 666, 'x2': 1438, 'y2': 960},
    "importer_of_record": {'x1': 166, 'y1': 960, 'x2': 1438, 'y2': 1394},
    "ship_to": {'x1': 162, 'y1': 1385, 'x2': 1438, 'y2': 1740},
    "invoice_type": {'x1': 1438, 'y1': 670, 'x2': 2306, 'y2': 855},
    "reference_no": {'x1': 1438, 'y1': 859, 'x2': 1876, 'y2': 1048},
    "reference_date": {'x1': 1881, 'y1': 846, 'x2': 2302, 'y2': 1061},
    "po_no": {'x1': 1433, 'y1': 1043, 'x2': 1863, 'y2': 1433},
    "po_no_date": {'x1': 1850, 'y1': 1043, 'x2': 2302, 'y2': 1438},
    "port_of_loading": {'x1': 1438, 'y1': 1447, 'x2': 1885, 'y2': 1745},
    "port_of_discharge": {'x1': 1841, 'y1': 1429, 'x2': 2297, 'y2': 1736}
}

REGIONS = {
    "shipper": {'x1': 155, 'y1': 635, 'x2': 1380, 'y2': 917},
    "importer_of_record":  {'x1': 155, 'y1': 909, 'x2': 1380, 'y2': 1334},
    "ship_to": {'x1': 151, 'y1': 1334, 'x2': 1380, 'y2': 1675},
    "invoice_type": {'x1': 1388, 'y1': 639, 'x2': 2209, 'y2': 829},
    "reference_no": {'x1': 1384, 'y1': 824, 'x2': 1797, 'y2': 1001},
    "reference_date": {'x1': 1788, 'y1': 812, 'x2': 2213, 'y2': 1001},
    "po_no":{'x1': 1380, 'y1': 1014, 'x2': 1788, 'y2': 1380},
    "po_no_date": {'x1': 1792, 'y1': 1001, 'x2': 2213, 'y2': 1380},
    "port_of_loading": {'x1': 1384, 'y1': 1376, 'x2': 1809, 'y2': 1675},
    "port_of_discharge": {'x1': 1780, 'y1': 1367, 'x2': 2213, 'y2': 1666}
}

TABLE_HEADER_REGIONS = {
    "Item": {'x1': 162, 'y1': 1736, 'x2': 267, 'y2': 1837},
    "GPN ": {'x1': 263, 'y1': 1736, 'x2': 412, 'y2': 1837},
    "DESCRIPTION": {'x1': 412, 'y1': 1740, 'x2': 1438, 'y2': 1832},
    "COUNTRY OF ORIGIN": {'x1': 1433, 'y1': 1736, 'x2': 1626, 'y2': 1832},
    "UOM": {'x1': 1622, 'y1': 1740, 'x2': 1723, 'y2': 1832},
    "QUANTITY": {'x1': 1714, 'y1': 1740, 'x2': 1885, 'y2': 1832},
    "FOB PRICE SGD": {'x1': 1885, 'y1': 1736, 'x2': 2091, 'y2': 1832},
    "FOB AMOUNT SGD": {'x1': 2087, 'y1': 1732, 'x2': 2310, 'y2': 1828},
}


TABLE_HEADER_REGIONS = {
    "Item": {'x1': 147, 'y1': 1670, 'x2': 260, 'y2': 1763},
    "GPN ": {'x1': 256, 'y1': 1662, 'x2': 395, 'y2': 1763},
    "DESCRIPTION": {'x1': 404, 'y1': 1654, 'x2': 1384, 'y2': 1763},
    "COUNTRY OF ORIGIN": {'x1': 1388, 'y1': 1658, 'x2': 1553, 'y2': 1755},
    "UOM": {'x1': 1548, 'y1': 1666, 'x2': 1654, 'y2': 1759},
    "QUANTITY": {'x1': 1658, 'y1': 1670, 'x2': 1813, 'y2': 1755},
    "FOB PRICE SGD": {'x1': 1809, 'y1': 1662, 'x2': 2003, 'y2': 1763},
    "FOB AMOUNT SGD": {'x1': 2003, 'y1': 1666, 'x2': 2213, 'y2': 1759},
}

def is_inside_region(box, region):
    # (unchanged – this is pure calculation, no I/O)
    box_x1 = box["x"]
    box_y1 = box["y"]
    box_x2 = box["x"] + box["width"]
    box_y2 = box["y"] + box["height"]

    reg_x1 = region["x1"]
    reg_y1 = region["y1"]
    reg_x2 = region["x2"]
    reg_y2 = region["y2"]

    x_overlap = not (box_x2 < reg_x1 or box_x1 > reg_x2)
    if not x_overlap:
        return False

    y1_inside = reg_y1 <= box_y1 <= reg_y2
    y2_inside = reg_y1 <= box_y2 <= reg_y2

    if y1_inside and not y2_inside:
        return False
    if y2_inside and not y1_inside:
        return True
    if y1_inside and y2_inside:
        return True
    return False


# ──────────────────────────────────────────────
#  ASYNC versions of extraction functions
# ──────────────────────────────────────────────

async def extract_fields_from_ocr(ocr_response, regions):
    def sync_extract():
        extracted_data = {}

        for field, region in regions.items():
            matched_boxes = [
                box for box in ocr_response
                if is_inside_region(box, region)
            ]
            matched_boxes.sort(key=lambda b: b["y"])
            field_text = "\n".join(b["text"] for b in matched_boxes)
            extracted_data[field] = field_text.strip()

        # Small post-processing
        ref_date = extracted_data.get('reference_date', '')
        if ref_date.endswith('Date:'):
            extracted_data['reference_date'] = 'Date: ' + ref_date.replace('\nDate:', '').strip()

        po_date = extracted_data.get('po_no_date', '')
        if po_date.endswith('Date:'):
            extracted_data['po_no_date'] = 'Date: ' + po_date.replace('\nDate:', '').strip()

        return extracted_data

    return await run_in_threadpool(sync_extract)


async def extract_table_rows(ocr_results: List[Dict], header_regions: Dict):
    def inner_extract_table_rows():
        # ── the whole original extract_table_rows logic here ──
        # (I've only wrapped it – logic stays exactly the same)

        column_keys = {
            "Item": "item",
            "GPN ": "gpn",
            "DESCRIPTION": "description",
            "COUNTRY OF ORIGIN": "country_of_origin",
            "UOM": "uom",
            "QUANTITY": "quantity",
            "FOB PRICE SGD": "fob_price_sgd",
            "FOB AMOUNT SGD": "fob_amount_sgd"
        }

        header_y1s = [r['y1'] for r in header_regions.values()]
        header_y2s = [r['y2'] for r in header_regions.values()]
        mean_header_y1 = np.mean(header_y1s)
        mean_header_y2 = np.mean(header_y2s)
        max_header_y2 = max(header_y2s)

        table_end_y = None
        first_row= []
        for det in ocr_results:
            if det['text'].strip().upper() == 'TOTAL':
                table_end_y = det['y']
                break
            if re.fullmatch(r'ROW\s*\d+', det['text'].strip().upper()):
            
                first_row.append({'item': '', 'gpn': '', 'description': det['text'].strip().upper(), 'country_of_origin': '',
                               'uom': '', 'quantity': '', 'fob_price_sgd': '', 'fob_amount_sgd': ''})
        if table_end_y is None:
            table_end_y = float('inf')

        def is_in_column(det: Dict, col_region: Dict, start_y: float) -> bool:
            center_x = det['x'] + det['width'] / 2
            center_y = det['y'] + det['height'] / 2
            return (col_region['x1'] <= center_x <= col_region['x2'] and 
                    start_y <= center_y < table_end_y)

        column_texts: Dict[str, List[Tuple[float, str, Dict]]] = {key: [] for key in header_regions}
        for det in ocr_results:
            text = det['text'].strip()
            if not text:
                continue
            center_y = det['y'] + det['height'] / 2
            if center_y <= max_header_y2:
                continue
            for col_name, col_region in header_regions.items():
                if is_in_column(det, col_region, max_header_y2 + 1):
                    column_texts[col_name].append((det['y'], text, det))
                    break

        for col in column_texts:
            column_texts[col].sort(key=lambda t: t[0])

        single_line_cols = ["Item", "QUANTITY", "UOM", "GPN ", "COUNTRY OF ORIGIN", "FOB PRICE SGD", "FOB AMOUNT SGD"]
        num_rows = 0
        row_start_ys = []
        for col in single_line_cols:
            if col in column_texts and len(column_texts[col]) > num_rows:
                num_rows = len(column_texts[col])
                row_start_ys = [t[0] for t in column_texts[col]]

        if num_rows == 0:
            return [], []

        if num_rows > 1:
            heights = [row_start_ys[i+1] - row_start_ys[i] for i in range(num_rows-1)]
            avg_height = np.mean(heights)
            max_height = np.max(heights)
        else:
            avg_height = 100
            max_height = avg_height

        row_bboxes = []
        for i in range(num_rows):
            y1 = row_start_ys[i]
            y2 = row_start_ys[i+1] if i < num_rows - 1 else y1 + max_height
            row_bboxes.append({'y1': y1, 'y2': y2})

        rows = []
        desc_col = "DESCRIPTION"
        for row_idx, row_bbox in enumerate(row_bboxes):
            row_data = {val: "" for val in column_keys.values()}
            for col_name, target_key in column_keys.items():
                col_texts = []
                for y, text, det in column_texts[col_name]:
                    center_y = det['y'] + det['height'] / 2
                    if row_bbox['y1'] <= center_y < row_bbox['y2']:
                        col_texts.append(text)
                if col_name == desc_col:
                    row_data[target_key] = "\n".join(col_texts)
                else:
                    row_data[target_key] = " ".join(col_texts)
            rows.append(row_data)

        return rows, row_bboxes, first_row

    return await run_in_threadpool(inner_extract_table_rows)


async def extract_total_amount(ocr_results: List[Dict]) -> Optional[str]:
    def sync_extract_total():
        # (original logic unchanged)
        total_labels = []
        for det in ocr_results:
            text = det['text'].strip().upper()
            if 'TOTAL' in text:
                total_labels.append({
                    'text': det['text'].strip(),
                    'y': det['y'],
                    'x': det['x'],
                    'width': det['width'],
                    'height': det['height'],
                    'center_y': det['y'] + det['height']/2,
                    'center_x': det['x'] + det['width']/2
                })

        if not total_labels:
            return None

        main_total = total_labels[0]
        candidates = []
        TOLERANCE_Y = 40

        for det in ocr_results:
            text = det['text'].strip()
            if not text:
                continue
            center_y = det['y'] + det['height']/2
            center_x = det['x'] + det['width']/2
            if abs(center_y - main_total['center_y']) <= TOLERANCE_Y:
                if center_x > main_total['center_x'] + 20:
                    if any(c.isdigit() for c in text):
                        candidates.append((center_x, text, det))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return await run_in_threadpool(sync_extract_total)


async def clean_currency(value_str: str) -> Decimal:
    def sync_clean():
        if not value_str or value_str.strip() == '':
            return Decimal('0')
        cleaned = re.sub(r'[\$\s]', '', value_str.strip())
        cleaned = cleaned.replace(',', '')
        try:
            return Decimal(cleaned)
        except:
            return Decimal('0')

    return await run_in_threadpool(sync_clean)


async def format_currency(value: Decimal, decimals: int = 2) -> str:
    def sync_format():
        if decimals == 0:
            formatted = f"{value:,.0f}"
        else:
            formatted = f"{value:,.{decimals}f}"
        return f"{formatted}"

    return await run_in_threadpool(sync_format)


async def recalculate_prices(
    items: List[Dict[str, str]],
    dividing_by: Optional[float] = None,
    price_decimals: int = 2,
    amount_decimals: int = 2
) -> List[Dict[str, str]]:
    async def inner_recalculate():
        result = []
        new_total = 0
        for item in items:
            new_item = item.copy()
            
            if 'quantity' not in item or 'fob_price_sgd' not in item:
                result.append(new_item)
                continue
            if not item['quantity'] and not item['fob_price_sgd']:
                result.append(new_item)
                continue
            try:
                qty = Decimal(item['quantity'].replace(',', '').strip())
                original_price_str = item['fob_price_sgd']
                current_price = await clean_currency(original_price_str)

                if dividing_by is not None and dividing_by != 0:
                    new_price = current_price / Decimal(str(dividing_by))
                else:
                    new_price = current_price
                

                new_price = new_price.quantize(
                    Decimal('1.' + '0' * price_decimals),
                    rounding=ROUND_HALF_UP
                )

                new_amount = qty * new_price
                new_amount = new_amount.quantize(
                    Decimal('1.' + '0' * amount_decimals),
                    rounding=ROUND_HALF_UP
                )
                new_total += new_amount

                new_item['fob_price_sgd'] = await format_currency(new_price, price_decimals)
                new_item['fob_amount_sgd'] = await format_currency(new_amount, amount_decimals)

            except Exception:
                new_item['fob_price_sgd'] = original_price_str.strip()
                new_item['fob_amount_sgd'] = item.get('fob_amount_sgd', '').strip()

            result.append(new_item)
        return result, await format_currency(new_total)

    return await inner_recalculate()


# OCR functions (already async or unchanged)
async def extract_text_from_image(image: Image.Image):
    return await run_in_threadpool(extract_text_all_levels, image)

def get_data(df):
    shipper = df[df.iloc[:, 0].astype(str).str.contains('SHIPPER', case=False, na=False)].index[0]
    importer = df[df.iloc[:, 0].astype(str).str.contains('Importer of Record', case=False, na=False)].index[0]
    ship_to = df[df.iloc[:, 0].astype(str).str.contains('Ship to', case=False, na=False)].index[0]
    table_start = df[df.apply(lambda row: 'ITEM' in row.values and 'GPN' in row.values, axis=1)].index[0]
    port_loading = df[df.iloc[:, 5].astype(str).str.contains('Port of Loading', case=False, na=False)].index[0]
    po_no = df[df.iloc[:, 5].astype(str).str.contains('PO no:', case=False, na=False)].index[0]
    reference_no = df[df.iloc[:, 5].astype(str).str.contains('Reference No:', case=False, na=False)].index[0]
    end_rows = df[df.iloc[:, 0].astype(str).str.contains('TOTAL', case=False, na=False)].index[0]
    shipper_ = "\n".join(df.iloc[shipper : importer -1, 0])
    importer_ = "\n".join(df.iloc[importer : ship_to -1, 0])
    ship_to_ = "\n".join(df.iloc[ship_to : table_start -1, 0])
    port_loading_ = "\n".join(df.iloc[port_loading : port_loading +2, 5])
    port_discharge = "\n".join(df.iloc[port_loading : port_loading +2, 7])
    reference_no_ = "\n".join(df.iloc[reference_no : reference_no +1, 5])
    reference_no_date = "\n".join(df.iloc[reference_no : reference_no +1, 7])
    po_no_ = "\n".join(df.iloc[po_no : po_no +1, 5])
    po_no_date = "\n".join(df.iloc[po_no : po_no +1, 7])
    total = df.iloc[end_rows, 7]
    
    return shipper_, importer_, ship_to_, reference_no_, reference_no_date, port_loading_, port_discharge, po_no_, po_no_date, total

def get_table_items(df):
    header_row = df[df.apply(lambda row: 'ITEM' in row.values and 'GPN' in row.values and 'DESCRIPTION ' in row.values, axis=1)]

    if header_row.empty:
        raise ValueError("Could not find header row with 'ITEM', 'Description' and 'GPN'")

    header_idx = header_row.index[0]
    print(f"Header found at index: {header_idx}")

    # Find the end of the table: the row that contains "RINGGIT MALAYSIA" in column 0
    end_rows = df[df.iloc[:, 0].astype(str).str.contains('TOTAL', case=False, na=False)]

    if end_rows.empty:
        print("Warning: 'RINGGIT MALAYSIA' not found → using full data after header")
        end_idx = len(df)
    else:
        end_idx = end_rows.index[0]
        print(f"Table ends before index: {end_idx}")
    #  Extract only the transaction rows
    df_transactions = df.iloc[header_idx:end_idx].copy()

    # Set the first row (which is the header) as column names
    df_transactions.columns = df_transactions.iloc[0]   # Use the DATE, REF.NO. row as headers
    df_transactions = df_transactions[1:]               # Remove the header row from data
    df_transactions = df_transactions.reset_index(drop=True)

    # Clean up column names and drop completely empty columns
    df_transactions.columns = df_transactions.columns.fillna('').str.strip()
    df_transactions = df_transactions.loc[:, (df_transactions != "").any(axis=0)]  # remove empty cols
    df_transactions = df_transactions.dropna(how='all').reset_index(drop=True)  
    dic = df_transactions.to_dict(orient='records')
    return clean_dataframe(dic)



def clean_dataframe(dic):
    new_data = []
    for data in dic:
        gpn = data.get('GPN', '')
        # --- normalize GPN ---
        if pd.isna(gpn) or str(gpn).strip().lower() == 'nan':
            ref = ''
        else:
            ref = str(gpn).strip()
        # ref = str(gpn).strip() if gpn is not None else ''
       
        item = data.get('ITEM')
        desc = (data.get('DESCRIPTION') or '').strip()

        country = data.get('CONTRY OF ORIGIN')
        uom = data.get('UOM')
        qty = data.get('QUANTITY')
        fob_price = data.get('FOB PRICE SGD')
        fob_amount = data.get('FOB AMOUNT SGD')

        if re.fullmatch(r'ROW\s*\d+', desc):
            pass

        if pd.isnull(gpn) and pd.isnull(ref) or '' == ref  and desc and new_data:
            
            new_data[-1]['description'] += "\n" + desc
        
        elif pd.isnull(gpn) and ref and new_data:
            if desc:
                new_data[-1]['description'] += "\n" + desc
            # new_data[-1]['GPN'] += " \n * " + ref
        
            
        else:
            data_ = {
            "item": None if pd.isnull(item) else str(item), 
            'gpn': None if pd.isnull(ref) else None if pd.isna(ref) else ref,
            'description': None if pd.isnull(desc) else desc,
            'country_of_origin': None if pd.isnull(country) else country,
            'uom': None if pd.isnull(uom) else uom,
            'quantity': None if pd.isnull(qty) else f"{qty:,.{2}f}",
            'fob_price_sgd': None if pd.isnull(fob_price) else f"{fob_price:,.{2}f}",
            'fob_amount_sgd': None if pd.isnull(fob_amount) else f"{fob_amount:,.{2}f}"
        }
            new_data.append(data_)
    return new_data
import asyncio
async def run_field_extraction_sync(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


async def analysis_receipt(df):

    (
            (shipper, importer_of_record, ship_to, reference_no, reference_date, port_loading, port_discharge, po_no, po_no_date, total), 
            items
        ) = await asyncio.gather(
            run_field_extraction_sync(get_data, df),
            run_field_extraction_sync(get_table_items, df),
    
        )
    total = await clean_currency(str(total))
    total = await format_currency(total)
    data = dict(
                shipper = shipper,
                importer_of_record = importer_of_record,
                ship_to = ship_to,
                invoice_type = None,
                reference_no = reference_no,
                reference_date = reference_date,
                po_no = po_no,
                po_no_date = po_no_date,
                port_of_loading = port_loading,
                port_of_discharge = port_discharge,
                items = items,
                total = total
    )
    return data


def extract_text_all_levels(image):
    # (your original function – unchanged)
    try:
        result_data = ocr_engine.predict(input=image)[0]
        predictions = {"lines": [], "words": [], "characters": []}

        for poly, text, score, box_coords in zip(
            result_data['dt_polys'],
            result_data['rec_texts'],
            result_data['rec_scores'],
            result_data['rec_boxes']
        ):
            polygon = np.array(poly)
            x_min, x_max = polygon[:, 0].min(), polygon[:, 0].max()
            y_min, y_max = polygon[:, 1].min(), polygon[:, 1].max()

            line_entry = {
                "text": text,
                "x": float(x_min),
                "y": float(y_min),
                "width": float(x_max - x_min),
                "height": float(y_max - y_min),
                "rotation": 0,
                "score": float(score)
            }
            predictions["lines"].append(line_entry)

            # word & char approximation (unchanged)
            words = re.findall(r'\S+', text)
            word_count = len(words)
            if word_count > 0:
                word_width = line_entry["width"] / word_count
                for i, word in enumerate(words):
                    word_entry = {
                        "text": word,
                        "x": float(x_min + i * word_width),
                        "y": float(y_min),
                        "width": float(word_width),
                        "height": float(y_max - y_min),
                        "rotation": 0,
                        "score": float(score),
                        "parent_line": line_entry["text"]
                    }
                    predictions["words"].append(word_entry)

                    for j, char in enumerate(word):
                        char_width = word_width / len(word)
                        char_entry = {
                            "text": char,
                            "x": float(word_entry["x"] + j * char_width),
                            "y": float(y_min),
                            "width": float(char_width),
                            "height": float(y_max - y_min),
                            "rotation": 0,
                            "score": float(score),
                            "parent_word": word,
                            "parent_line": line_entry["text"]
                        }
                        predictions["characters"].append(char_entry)

        return predictions

    finally:
        paddle.device.cuda.empty_cache()


def convert_pdf_to_images_gfs(pdf_bytes, zoom=4):
    images = []
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            # Use raw RGB data directly for highest quality
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(image)

    return images

# ──────────────────────────────────────────────
#                  ENDPOINT
# ──────────────────────────────────────────────



def convert_pdf_to_images(pdf_bytes, zoom=4):
    """
    Convert PDF bytes to a list of PIL Image objects with consistent preprocessing.
    
    Args:
        pdf_bytes: Bytes object containing the PDF file.
        zoom: Zoom factor for resolution (default 4, equivalent to ~288 DPI).
    
    Returns:
        List of PIL Image objects in RGB format.
    """
    images = []
    matrix = fitz.Matrix(zoom, zoom)  # Zoom factor for high resolution
    
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)  # No alpha channel
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
    except Exception as e:
        logging.error(f"Error converting PDF to images: {str(e)}")
        raise
    
    return images


# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):

    return templates.TemplateResponse("invoice.html", {"request": request})

@app.post("/extractocr")
async def extract_ocr(request: Request):
    logging.info("Received request at /extractocr")
    form = await request.form()
    output_results = []

    for field_name, file in form.items():
        try:
            file_bytes = await file.read()
            file_name = file.filename or f"{uuid.uuid4()}"

            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf', '.xlsx',  '.xls')):
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {file_name}")
            
            


            image = None
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            elif file_name.lower().endswith('pdf'):
                images = convert_pdf_to_images_gfs(file_bytes)
                if len(images) == 1:
                    image = images[0]
                else:
                    print('multiple images')
                    for i, image in enumerate(images):
                        pass

            if image:
                image_np = np.array(image)

                ocr_results = await extract_text_from_image(image_np)

                # All extraction steps now awaited
                extracted_text = await extract_fields_from_ocr(ocr_results["lines"], REGIONS)

                table_rows, row_bboxes, first_row = await extract_table_rows(ocr_results["lines"], TABLE_HEADER_REGIONS)

                cleaned_items, new_total = await recalculate_prices(
                    [row for row in table_rows if any(row.values())]
                )
                extracted_text['items'] = cleaned_items
                if first_row:
                    # extracted_text['items'].insert(0, first_row[0])
                    first_row.extend(cleaned_items)
                    extracted_text['items'] = first_row

                total = await extract_total_amount(ocr_results["lines"])
                if total:
                    total = await clean_currency(total)
                    total = await format_currency(total)
                    extracted_text['total'] =total

                output_results.append({
                    "filename": file_name,
                    "responses": extracted_text
                })

            if  file_name.lower().endswith(('.xlsx', '.xls')):
                file_io = io.BytesIO(file_bytes)

                if file_name.lower().endswith('.xlsx'):
                    df = pd.read_excel(file_io, engine='openpyxl')
                else:
                    df = pd.read_excel(file_io, engine='xlrd')

                if df.empty:
                    output_results.append({
                        "filename": file_name,
                        "responses": {"error": "Empty Excel file"}
                    })
                    continue

                result = await analysis_receipt(df)
                df.columns = df.columns.str.strip()
                # clean_result = jsonable_encoder(result)

                output_results.append({
                    "filename": file_name,
                    "responses": result
                })

        except Exception as e:
            logging.error(f"Error processing file '{file_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file '{file_name}': {e}")

    return JSONResponse(content=output_results)



@app.post("/recalculate")
async def recalculate_unit_price(payload: RecalculateRequest):
    logging.info("Received request at /recalculate-unit-price")

    try:
        if payload.dividing_by <= 0:
            raise HTTPException(
                status_code=400,
                detail="dividing_by must be greater than 0"
            )

        # Convert Pydantic models to dicts
        items = [item.dict() for item in payload.items]

        # Call your existing function
        updated_items, new_total = await recalculate_prices(
            items,
            dividing_by=payload.dividing_by
        )

        return JSONResponse(content={
            "dividing_by": payload.dividing_by,
            "items": updated_items,
            'total': new_total
        })

    except HTTPException:
        raise

    except Exception as e:
        logging.error(f"Error recalculating prices: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error recalculating prices: {e}"
        )




