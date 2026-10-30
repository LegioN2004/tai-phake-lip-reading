import xml.etree.ElementTree as ET
import os

def build_drawio():
    mxfile = ET.Element('mxfile', {
        'host': 'app.diagrams.net',
        'modified': '2026-10-29T06:00:00.000Z',
        'agent': 'Antigravity IEEE Visual Speech Recognition Diagram Generator',
        'version': '24.7.17'
    })

    # =========================================================================
    # STYLING DEFINITIONS (Flat, Monochrome-Plus-One-Accent Journal Style)
    # Background: Pure solid white (#FFFFFF)
    # Neutral: Crisp Slate (#334155 / #FFFFFF / #F8FAFC)
    # Accent: Academic Cobalt Blue (#1D4ED8 / #EFF6FF / #BFDBFE)
    # =========================================================================

    # Explicit white canvas background rect
    STYLE_CANVAS_BG = (
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=none;"
    )

    # Row container (Horizontal lane)
    STYLE_ROW_CONTAINER = (
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1.2;"
    )

    # Model Name Pill / Card
    STYLE_MODEL_TITLE_CARD = (
        "rounded=0;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=11.5;fontStyle=1;"
        "fontColor=#0F172A;strokeColor=#1E293B;strokeWidth=1.6;fillColor=#F1F5F9;"
        "align=center;verticalAlign=middle;"
    )

    # Dashed background panel for temporal module
    STYLE_TEMPORAL_BG_PANEL = (
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#93C5FD;strokeWidth=1.4;dashed=1;"
        "verticalAlign=top;align=left;spacingLeft=10;spacingTop=4;fontFamily=Helvetica;fontSize=8.5;fontStyle=1;fontColor=#1D4ED8;"
    )

    # Standard / Shared node style (Clean pure white box, dark slate border)
    STYLE_SHARED_NODE = (
        "rounded=0;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=10;"
        "fontColor=#0F172A;strokeColor=#334155;strokeWidth=1.2;fillColor=#FFFFFF;"
        "align=center;verticalAlign=middle;"
    )

    # ACCENT COLOR: Middle Differing Temporal Modules (Cobalt Blue border, soft blue tint)
    STYLE_ACCENT_NODE = (
        "rounded=0;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=10;"
        "fontColor=#1E3A8A;strokeColor=#1D4ED8;strokeWidth=1.8;fillColor=#EFF6FF;"
        "align=center;verticalAlign=middle;"
    )

    # Total parameters badge (Right side of each row)
    STYLE_TOTAL_PARAM_BADGE = (
        "rounded=0;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=10;fontStyle=1;"
        "fontColor=#0F172A;strokeColor=#334155;strokeWidth=1.4;fillColor=#F1F5F9;"
        "align=center;verticalAlign=middle;"
    )

    # Output node
    STYLE_OUTPUT_NODE = (
        "rounded=0;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=10;fontStyle=1;"
        "fontColor=#0F172A;strokeColor=#0F172A;strokeWidth=1.5;fillColor=#FFFFFF;"
        "align=center;verticalAlign=middle;"
    )

    # Loss objective box
    STYLE_LOSS_BOX = (
        "rounded=0;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=10.5;fontStyle=1;"
        "fontColor=#0F172A;strokeColor=#475569;strokeWidth=1.2;fillColor=#FFFFFF;"
        "align=center;verticalAlign=middle;"
    )

    # Horizontal connector edge (Left to Right)
    STYLE_EDGE = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=1;whiteSpace=wrap;strokeColor=#475569;strokeWidth=1.2;fontFamily=Helvetica;fontSize=8;"
        "fontColor=#334155;fontStyle=2;labelBackgroundColor=#FFFFFF;labelBorderColor=none;"
        "endArrow=classic;endSize=5;endFill=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
    )

    def create_cell(root, cid, val, x, y, w, h, style):
        c = ET.SubElement(root, 'mxCell', {
            'id': cid,
            'value': val,
            'style': style,
            'vertex': '1',
            'parent': '1'
        })
        ET.SubElement(c, 'mxGeometry', {
            'x': str(x),
            'y': str(y),
            'width': str(w),
            'height': str(h),
            'as': 'geometry'
        })
        return c

    def add_edge(root, eid, value, source, target, style=STYLE_EDGE):
        edge = ET.SubElement(root, 'mxCell', {
            'id': eid,
            'value': value,
            'style': style,
            'edge': '1',
            'parent': '1',
            'source': source,
            'target': target
        })
        ET.SubElement(edge, 'mxGeometry', {'relative': '1', 'as': 'geometry'})
        return edge

    # =========================================================================
    # DIAGRAM PAGE: Horizontal Layout (Three Rows, Pure White Background)
    # =========================================================================
    diag = ET.SubElement(mxfile, 'diagram', {
        'id': 'page_horizontal_comparison',
        'name': 'Horizontal VSR Comparison'
    })
    model = ET.SubElement(diag, 'mxGraphModel', {
        'dx': '1500', 'dy': '900', 'grid': '1', 'gridSize': '10', 'guides': '1',
        'tooltips': '1', 'connect': '1', 'arrows': '1', 'fold': '1', 'page': '1',
        'pageScale': '1', 'pageWidth': '2100', 'pageHeight': '720',
        'background': '#FFFFFF', 'math': '0', 'shadow': '0'
    })
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', {'id': '0'})
    ET.SubElement(root, 'mxCell', {'id': '1', 'parent': '0'})

    # 1. Guaranteed White Background Rect (eliminates any dark mode/transparent canvas artifacts)
    create_cell(root, 'canvas_white_bg', '', 0, 0, 2100, 720, STYLE_CANVAS_BG)

    # 2. Figure Header Title
    create_cell(root, 'fig_title',
                '<b>COMPARATIVE ARCHITECTURES FOR TAI PHAKE VISUAL SPEECH RECOGNITION</b><br>'
                '<font style="font-size: 11px; font-weight: normal; color: #475569;">'
                'Three independent end-to-end pipelines sharing identical spatial front-ends and dense classification heads; temporal modules visually highlighted as the only architectural difference.'
                '</font>',
                40, 15, 2020, 45,
                "text;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=13;fontColor=#0F172A;")

    # Model specifications definition
    models_config = [
        {
            'id': 'A',
            'plain_title': 'Model A<br>(BiLSTM)',
            'row_y': 75,
            'total_params_str': 'Total Parameters:<br><b>6.75 M</b>',
            'temporal_label': '1.44 M params',
            'temporal_box_title': '1-layer BiLSTM',
            'temporal_box_detail': '128 units/dir (256 total), input_dim=1280  ⇄',
            'temporal_state_title': 'Last Hidden State<br>(256-D)',
            'dropout_val': 'Dropout<br>(rate = 0.4)',
        },
        {
            'id': 'B',
            'plain_title': 'Model B<br>(BiGRU)',
            'row_y': 235,
            'total_params_str': 'Total Parameters:<br><b>6.40 M</b>',
            'temporal_label': '1.08 M params',
            'temporal_box_title': '1-layer BiGRU',
            'temporal_box_detail': '128 units/dir (256 total), input_dim=1280  ⇄',
            'temporal_state_title': 'Last Hidden State<br>(256-D)',
            'dropout_val': 'Dropout<br>(rate = 0.4)',
        },
        {
            'id': 'C',
            'plain_title': 'Model C<br>(Transformer)',
            'row_y': 395,
            'total_params_str': 'Total Parameters:<br><b>6.90 M</b>',
            'temporal_label': '1.58 M params',
            'temporal_box_title': 'Transformer Encoder ×2',
            'temporal_box_detail': 'd_model=256, 4 heads, FFN=1024, Pre-LN',
            'temporal_state_title': 'Mean Pool<br>(30 tokens)',
            'dropout_val': 'Dropout 0.1 (Attn/FFN)<br>+ 0.2 (Head)',
        }
    ]

    for m in models_config:
        mid = m['id']
        ry = m['row_y']
        by = ry + 42  # block center Y inside row
        bh = 62       # standard block height

        # Row Container (Pure white background, crisp slate border)
        create_cell(root, f'row_{mid}', '', 40, ry, 2020, 145, STYLE_ROW_CONTAINER)

        # Model Name Card (Leftmost block)
        create_cell(root, f'card_title_{mid}', m['plain_title'], 55, by, 110, bh, STYLE_MODEL_TITLE_CARD)

        # ---------------------------------------------------------------------
        # SHARED SPATIAL FRONT-END (Stages 1 to 4)
        # ---------------------------------------------------------------------
        # 1. Input Video Tensor
        create_cell(root, f'top_input_{mid}',
                    '<b>Input Video Tensor</b><br>'
                    '<font style="font-size: 8.5px; color: #64748B;">(Batch, 30, 128, 128, 3)<br>3-Ch RGB</font>',
                    185, by, 125, bh, STYLE_SHARED_NODE)

        # 2. Frame-level EfficientNet-B0
        create_cell(root, f'top_effnet_{mid}',
                    '<b>EfficientNet-B0</b><br>'
                    '<font style="font-size: 8.5px; color: #475569;">Frame-level<br>ImageNet, frozen BN</font>',
                    330, by, 125, bh, STYLE_SHARED_NODE)

        # 3. Global Average Pool
        create_cell(root, f'top_gap_{mid}',
                    '<b>Global Average Pool</b><br>'
                    '<font style="font-size: 8.5px; color: #475569;">1280-D per frame<br>(Batch, 30, 1280)</font>',
                    475, by, 120, bh, STYLE_SHARED_NODE)

        # 4. Sequence Tensor
        create_cell(root, f'top_seq_{mid}',
                    '<b>Sequence Tensor</b><br>'
                    '<font style="font-size: 8.5px; font-weight: bold; color: #0F172A;">30 feature vectors<br>(Batch, 30, 1280)</font>',
                    615, by, 125, bh, STYLE_SHARED_NODE)

        # Front-End Edges
        add_edge(root, f'e_top_1_{mid}', '(Batch, 30, 128, 128, 3)', f'top_input_{mid}', f'top_effnet_{mid}')
        add_edge(root, f'e_top_2_{mid}', '(Batch, 30, 1280)', f'top_effnet_{mid}', f'top_gap_{mid}')
        add_edge(root, f'e_top_3_{mid}', '(Batch, 30, 1280)', f'top_gap_{mid}', f'top_seq_{mid}')

        # ---------------------------------------------------------------------
        # TEMPORAL MODULE (Differing Component - Accent Highlighted)
        # ---------------------------------------------------------------------
        if mid in ['A', 'B']:
            # Models A & B: Dense projection deleted.
            # Sequence Tensor connects directly to Temporal Module.
            create_cell(root, f'bg_mid_{mid}', f'TEMPORAL MODULE (DIFFERING COMPONENT  |  {m["temporal_label"]})',
                        765, ry + 12, 450, 120, STYLE_TEMPORAL_BG_PANEL)

            create_cell(root, f'mid_core_{mid}',
                        f"<b>{m['temporal_box_title']}</b><br>"
                        f"<font style=\"font-size: 8.5px; color: #1E3A8A;\">{m['temporal_box_detail']}</font><br>"
                        f"<font style=\"font-size: 8.5px; font-weight: bold; color: #1D4ED8;\">[ {m['temporal_label']} ]</font>",
                        785, by, 190, bh, STYLE_ACCENT_NODE)

            create_cell(root, f'mid_state_{mid}',
                        f"<b>{m['temporal_state_title']}</b><br>"
                        '<font style="font-size: 8.5px; color: #1E3A8A;">(Batch, 256)</font>',
                        1035, by, 150, bh, STYLE_ACCENT_NODE)

            add_edge(root, f'e_mid_in_{mid}', '(Batch, 30, 1280)', f'top_seq_{mid}', f'mid_core_{mid}')
            add_edge(root, f'e_mid_core_to_state_{mid}', '(Batch, 30, 256)', f'mid_core_{mid}', f'mid_state_{mid}')
            add_edge(root, f'e_mid_out_{mid}', '(Batch, 256)', f'mid_state_{mid}', f'bot_drop_{mid}')

        else:
            # Model C: Input Projection (1280 → 256) for d_model + Pos Enc + Transf x2 + Mean Pool
            create_cell(root, f'top_proj_{mid}',
                        '<b>Input Projection</b><br>'
                        '<font style="font-size: 8px; color: #475569;">1280 &rarr; 256 for d_model<br>(Batch, 30, 256)</font>',
                        760, by, 125, bh, STYLE_SHARED_NODE)

            add_edge(root, f'e_top_4_{mid}', '(Batch, 30, 1280)', f'top_seq_{mid}', f'top_proj_{mid}')

            create_cell(root, f'bg_mid_{mid}', f'TEMPORAL MODULE (DIFFERING COMPONENT  |  {m["temporal_label"]})',
                        905, ry + 12, 400, 120, STYLE_TEMPORAL_BG_PANEL)

            create_cell(root, f'mid_pos_{mid}',
                        '<b>Positional Encoding</b><br>'
                        '<font style="font-size: 8px; color: #1E3A8A;">Learnable (L=30)<br>(Batch, 30, 256)</font>',
                        920, by, 115, bh, STYLE_ACCENT_NODE)

            create_cell(root, f'mid_trans_{mid}',
                        f"<b>{m['temporal_box_title']}</b><br>"
                        f"<font style=\"font-size: 8px; color: #1E3A8A;\">{m['temporal_box_detail']}</font><br>"
                        f"<font style=\"font-size: 8px; font-weight: bold; color: #1D4ED8;\">[ {m['temporal_label']} ]</font>",
                        1055, by, 130, bh, STYLE_ACCENT_NODE)

            create_cell(root, f'mid_pool_{mid}',
                        f"<b>{m['temporal_state_title']}</b><br>"
                        '<font style="font-size: 8px; color: #1E3A8A;">(Batch, 256)</font>',
                        1205, by, 90, bh, STYLE_ACCENT_NODE)

            add_edge(root, f'e_mid_in_{mid}', '(Batch, 30, 256)', f'top_proj_{mid}', f'mid_pos_{mid}')
            add_edge(root, f'e_mid_pos_to_trans_{mid}', '(Batch, 30, 256)', f'mid_pos_{mid}', f'mid_trans_{mid}')
            add_edge(root, f'e_mid_trans_to_pool_{mid}', '(Batch, 30, 256)', f'mid_trans_{mid}', f'mid_pool_{mid}')
            add_edge(root, f'e_mid_out_{mid}', '(Batch, 256)', f'mid_pool_{mid}', f'bot_drop_{mid}')

        # ---------------------------------------------------------------------
        # SHARED CLASSIFICATION HEAD (Stages 7 to 10)
        # ---------------------------------------------------------------------
        # 7. Dropout
        create_cell(root, f'bot_drop_{mid}',
                    f"<b>{m['dropout_val']}</b>",
                    1325, by, 125, bh, STYLE_SHARED_NODE)

        # 8. FC 256 -> 11
        create_cell(root, f'bot_fc_{mid}',
                    '<b>FC 256 &rarr; 11</b><br>'
                    '<font style="font-size: 8px; color: #475569;">Dense layer<br>(2,827 params)</font>',
                    1475, by, 95, bh, STYLE_SHARED_NODE)

        # 9. Softmax
        create_cell(root, f'bot_softmax_{mid}',
                    '<b>Softmax</b><br>'
                    '<font style="font-size: 8px; color: #475569;">Posterior prob.<br>11 classes</font>',
                    1595, by, 85, bh, STYLE_SHARED_NODE)

        # 10. 11 Classes (Tai Phake Digits)
        create_cell(root, f'bot_out_{mid}',
                    '<b>11 classes</b><br>'
                    '<font style="font-size: 8px; color: #0F172A;">Tai Phake digits<br>d0 (Pau) … d10 (Sip)</font>',
                    1705, by, 115, bh, STYLE_OUTPUT_NODE)

        # Head Edges
        add_edge(root, f'e_bot_1_{mid}', '(Batch, 256)', f'bot_drop_{mid}', f'bot_fc_{mid}')
        add_edge(root, f'e_bot_2_{mid}', '(Batch, 11)', f'bot_fc_{mid}', f'bot_softmax_{mid}')
        add_edge(root, f'e_bot_3_{mid}', '(Batch, 11)', f'bot_softmax_{mid}', f'bot_out_{mid}')

        # Summary Pill (Rightmost)
        create_cell(root, f'summary_bottom_{mid}',
                    m['total_params_str'],
                    1845, by, 115, bh, STYLE_TOTAL_PARAM_BADGE)

    # Global Training Objective Box
    create_cell(root, 'loss_objective',
                '<b>Training Objective:</b> Class-weighted Cross-Entropy Loss + Label Smoothing (ε=0.1)',
                40, 560, 2020, 32, STYLE_LOSS_BOX)

    # Global Bottom Footnote
    create_cell(root, 'fig_legend',
                '<i>Note: All three architectures share an identical training recipe (AdamW, Cosine Annealing, 5-fold speaker-independent GroupKFold, 3 seeds/fold). The highlighted temporal block is visually and empirically the only architectural difference.</i>',
                40, 605, 2020, 30,
                "text;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=10.5;fontColor=#475569;")

    # Save to XML
    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ", level=0)

    out_docs = 'docs/figures/vsr_architectures_comparison.drawio'
    tree.write(out_docs, encoding='utf-8', xml_declaration=True)
    out_root = 'vsr_architectures_comparison.drawio'
    tree.write(out_root, encoding='utf-8', xml_declaration=True)
    print("Files successfully generated:")
    print(" -", out_docs)
    print(" -", out_root)

if __name__ == '__main__':
    build_drawio()
