import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const rootDir = path.resolve(import.meta.dirname, "..");
const outDir = path.join(rootDir, "subgraphs");

function coreProps(type) {
  return {
    ue_properties: {
      widget_ue_connectable: {},
      version: "7.7",
      input_ue_unconnectable: {},
    },
    cnr_id: "comfy-core",
    ver: "0.18.1",
    "Node name for S&R": type,
  };
}

function input(name, type, link = null, extra = {}) {
  return {
    localized_name: extra.localized_name ?? name,
    name,
    type,
    ...(extra.label ? { label: extra.label } : {}),
    ...(extra.shape ? { shape: extra.shape } : {}),
    ...(extra.widget ? { widget: extra.widget } : {}),
    link,
  };
}

function output(name, type, links = null, extra = {}) {
  return {
    localized_name: extra.localized_name ?? name,
    name,
    type,
    links,
    ...(extra.label ? { label: extra.label } : {}),
  };
}

function stableUuid(label) {
  const hash = crypto.createHash("sha1").update(label).digest("hex");
  return [
    hash.slice(0, 8),
    hash.slice(8, 12),
    `5${hash.slice(13, 16)}`,
    `${(Number.parseInt(hash.slice(16, 18), 16) & 0x3f | 0x80).toString(16)}${hash.slice(18, 20)}`,
    hash.slice(20, 32),
  ].join("-");
}

function makeBlueprint(gridSize) {
  const subgraphId = stableUuid(`crop-images-${gridSize}x${gridSize}`);
  let nextNodeId = 1;
  let nextLinkId = 1;
  const nodes = [];
  const links = [];
  const inputLinkIds = [];

  function addLink(origin_id, origin_slot, target_id, target_slot, type) {
    const id = nextLinkId++;
    links.push({ id, origin_id, origin_slot, target_id, target_slot, type });
    return id;
  }

  function nodeById(id) {
    return nodes.find((node) => node.id === id);
  }

  function connectInt(originNodeId, targetNodeId, targetSlot) {
    const link = addLink(originNodeId, 1, targetNodeId, targetSlot, "INT");
    nodeById(originNodeId).outputs[1].links.push(link);
    return link;
  }

  const getSizeId = nextNodeId++;
  const gridSizeId = nextNodeId++;
  const widthId = nextNodeId++;
  const heightId = nextNodeId++;

  nodes.push({
    id: getSizeId,
    type: "GetImageSize",
    pos: [0, 280],
    size: [230, 120],
    flags: {},
    order: 1,
    mode: 0,
    inputs: [input("image", "IMAGE")],
    outputs: [
      output("width", "INT", []),
      output("height", "INT", []),
      output("batch_size", "INT", null),
    ],
    properties: coreProps("GetImageSize"),
  });
  nodes.push({
    id: gridSizeId,
    type: "PrimitiveInt",
    pos: [0, 450],
    size: [230, 110],
    flags: {},
    order: 0,
    mode: 0,
    inputs: [input("value", "INT", null, { widget: { name: "value" } })],
    outputs: [output("INT", "INT", [])],
    properties: coreProps("PrimitiveInt"),
    widgets_values: [gridSize, "fixed"],
  });

  const imageToSize = addLink(-10, 0, getSizeId, 0, "IMAGE");
  inputLinkIds.push(imageToSize);
  nodeById(getSizeId).inputs[0].link = imageToSize;

  const widthInput = addLink(getSizeId, 0, widthId, 0, "INT");
  const gridToWidth = addLink(gridSizeId, 0, widthId, 1, "INT");
  const heightInput = addLink(getSizeId, 1, heightId, 0, "INT");
  const gridToHeight = addLink(gridSizeId, 0, heightId, 1, "INT");
  nodeById(getSizeId).outputs[0].links.push(widthInput);
  nodeById(getSizeId).outputs[1].links.push(heightInput);
  nodeById(gridSizeId).outputs[0].links.push(gridToWidth, gridToHeight);

  for (const spec of [
    [widthId, "Math Expression (Width)", [310, 180], widthInput, gridToWidth],
    [heightId, "Math Expression (Height)", [310, 430], heightInput, gridToHeight],
  ]) {
    const [id, title, pos, aLink, bLink] = spec;
    nodes.push({
      id,
      type: "ComfyMathExpression",
      pos,
      size: [370, 190],
      flags: {},
      order: id,
      mode: 0,
      inputs: [
        input("values.a", "FLOAT,INT", aLink, { label: "a", localized_name: "values.a" }),
        input("values.b", "FLOAT,INT", bLink, { label: "b", localized_name: "values.b", shape: 7 }),
        input("values.c", "FLOAT,INT", null, { label: "c", localized_name: "values.c", shape: 7 }),
        input("expression", "STRING", null, { widget: { name: "expression" } }),
      ],
      outputs: [output("FLOAT", "FLOAT", null), output("INT", "INT", [])],
      title,
      properties: coreProps("ComfyMathExpression"),
      widgets_values: ["max(1, int(a/b))"],
    });
  }

  const xStartNodeIds = [null, widthId];
  const yStartNodeIds = [null, heightId];
  for (let i = 2; i <= gridSize - 1; i++) {
    const id = nextNodeId++;
    const inLink = connectInt(widthId, id, 0);
    nodes.push({
      id,
      type: "ComfyMathExpression",
      pos: [760, 60 + i * 130],
      size: [300, 160],
      flags: {},
      order: id,
      mode: 0,
      inputs: [
        input("values.a", "FLOAT,INT", inLink, { label: "a", localized_name: "values.a" }),
        input("values.b", "FLOAT,INT", null, { label: "b", localized_name: "values.b", shape: 7 }),
        input("values.c", "FLOAT,INT", null, { label: "c", localized_name: "values.c", shape: 7 }),
        input("expression", "STRING", null, { widget: { name: "expression" } }),
      ],
      outputs: [output("FLOAT", "FLOAT", null), output("INT", "INT", [])],
      title: `Math Expression (x${i})`,
      properties: coreProps("ComfyMathExpression"),
      widgets_values: [`${i} * a`],
    });
    xStartNodeIds[i] = id;
  }
  for (let i = 2; i <= gridSize - 1; i++) {
    const id = nextNodeId++;
    const inLink = connectInt(heightId, id, 0);
    nodes.push({
      id,
      type: "ComfyMathExpression",
      pos: [760, 440 + i * 130],
      size: [300, 160],
      flags: {},
      order: id,
      mode: 0,
      inputs: [
        input("values.a", "FLOAT,INT", inLink, { label: "a", localized_name: "values.a" }),
        input("values.b", "FLOAT,INT", null, { label: "b", localized_name: "values.b", shape: 7 }),
        input("values.c", "FLOAT,INT", null, { label: "c", localized_name: "values.c", shape: 7 }),
        input("expression", "STRING", null, { widget: { name: "expression" } }),
      ],
      outputs: [output("FLOAT", "FLOAT", null), output("INT", "INT", [])],
      title: `Math Expression (y${i})`,
      properties: coreProps("ComfyMathExpression"),
      widgets_values: [`${i} * a`],
    });
    yStartNodeIds[i] = id;
  }

  const rightWidthId = nextNodeId++;
  const bottomHeightId = nextNodeId++;
  const fullWidthLink = addLink(getSizeId, 0, rightWidthId, 0, "INT");
  const rightStartLink = connectInt(xStartNodeIds[gridSize - 1], rightWidthId, 1);
  const fullHeightLink = addLink(getSizeId, 1, bottomHeightId, 0, "INT");
  const bottomStartLink = connectInt(yStartNodeIds[gridSize - 1], bottomHeightId, 1);
  nodeById(getSizeId).outputs[0].links.push(fullWidthLink);
  nodeById(getSizeId).outputs[1].links.push(fullHeightLink);

  for (const spec of [
    [rightWidthId, "Math Expression (Right Width)", [1120, 760], fullWidthLink, rightStartLink],
    [bottomHeightId, "Math Expression (Bottom Height)", [1120, 960], fullHeightLink, bottomStartLink],
  ]) {
    const [id, title, pos, aLink, bLink] = spec;
    nodes.push({
      id,
      type: "ComfyMathExpression",
      pos,
      size: [370, 190],
      flags: {},
      order: id,
      mode: 0,
      inputs: [
        input("values.a", "FLOAT,INT", aLink, { label: "a", localized_name: "values.a" }),
        input("values.b", "FLOAT,INT", bLink, { label: "b", localized_name: "values.b", shape: 7 }),
        input("values.c", "FLOAT,INT", null, { label: "c", localized_name: "values.c", shape: 7 }),
        input("expression", "STRING", null, { widget: { name: "expression" } }),
      ],
      outputs: [output("FLOAT", "FLOAT", null), output("INT", "INT", [])],
      title,
      properties: coreProps("ComfyMathExpression"),
      widgets_values: ["max(1, a - b)"],
    });
  }

  const cropIds = [];
  const individualOutputLinkIds = [];
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const bboxId = nextNodeId++;
      const cropId = nextNodeId++;
      const index = row * gridSize + col;
      cropIds.push(cropId);

      const bboxInputs = [
        input("x", "INT", null, { widget: { name: "x" } }),
        input("y", "INT", null, { widget: { name: "y" } }),
        input("width", "INT", null, { widget: { name: "width" } }),
        input("height", "INT", null, { widget: { name: "height" } }),
      ];
      if (col > 0) bboxInputs[0].link = connectInt(xStartNodeIds[col], bboxId, 0);
      if (row > 0) bboxInputs[1].link = connectInt(yStartNodeIds[row], bboxId, 1);
      bboxInputs[2].link = connectInt(col === gridSize - 1 ? rightWidthId : widthId, bboxId, 2);
      bboxInputs[3].link = connectInt(row === gridSize - 1 ? bottomHeightId : heightId, bboxId, 3);

      const bboxToCrop = addLink(bboxId, 0, cropId, 1, "BOUNDING_BOX");
      const imageToCrop = addLink(-10, 0, cropId, 0, "IMAGE");
      inputLinkIds.push(imageToCrop);
      nodes.push({
        id: bboxId,
        type: "PrimitiveBoundingBox",
        pos: [1540 + col * 360, 190 + row * 560],
        size: [270, 200],
        flags: {},
        order: bboxId,
        mode: 0,
        inputs: bboxInputs,
        outputs: [output("BOUNDING_BOX", "BOUNDING_BOX", [bboxToCrop])],
        properties: coreProps("PrimitiveBoundingBox"),
        widgets_values: [0, 0, 512, 512],
      });
      nodes.push({
        id: cropId,
        type: "ImageCropV2",
        pos: [1860 + col * 360, 60 + row * 560],
        size: [300, 480],
        flags: {},
        order: cropId,
        mode: 0,
        inputs: [
          input("image", "IMAGE", imageToCrop),
          input("crop_region", "BOUNDING_BOX", bboxToCrop, { widget: { name: "crop_region" } }),
        ],
        outputs: [output("IMAGE", "IMAGE", [])],
        properties: coreProps("ImageCropV2"),
        widgets_values: [{ x: 0, y: 0, width: 512, height: 512 }, 0, 0, 512, 512],
      });

      const outLink = addLink(cropId, 0, -20, index, "IMAGE");
      nodeById(cropId).outputs[0].links.push(outLink);
      individualOutputLinkIds[index] = outLink;
    }
  }

  const batchId = nextNodeId++;
  const batchInputs = cropIds.map((cropId, index) => {
    const link = addLink(cropId, 0, batchId, index, "IMAGE");
    nodeById(cropId).outputs[0].links.push(link);
    return input(`images.image${index}`, "IMAGE", link, {
      label: `image${index}`,
      localized_name: `images.image${index}`,
      ...(index >= 2 ? { shape: 7 } : {}),
    });
  });
  batchInputs.push(input(`images.image${cropIds.length}`, "IMAGE", null, {
    label: `image${cropIds.length}`,
    localized_name: `images.image${cropIds.length}`,
    shape: 7,
  }));
  const batchOutputLink = addLink(batchId, 0, -20, cropIds.length, "IMAGE");
  nodes.push({
    id: batchId,
    type: "BatchImagesNode",
    pos: [1540 + gridSize * 720, 240],
    size: [230, Math.max(290, 110 + gridSize * gridSize * 20)],
    flags: {},
    order: batchId,
    mode: 0,
    inputs: batchInputs,
    outputs: [output("IMAGE", "IMAGE", [batchOutputLink])],
    properties: coreProps("BatchImagesNode"),
  });

  const subOutputs = [];
  const topOutputs = [];
  for (let index = 0; index < cropIds.length; index++) {
    const row = Math.floor(index / gridSize) + 1;
    const col = (index % gridSize) + 1;
    const name = index === 0 ? "IMAGE" : `IMAGE_${index}`;
    const label = `row_${row}_col_${col}`;
    subOutputs.push({
      id: stableUuid(`crop-images-${gridSize}x${gridSize}-output-${index}`),
      name,
      type: "IMAGE",
      linkIds: [individualOutputLinkIds[index]],
      localized_name: name,
      label,
      pos: [1540 + gridSize * 760 + 280, 80 + index * 20],
    });
    topOutputs.push({ label, localized_name: name, name, type: "IMAGE", links: [] });
  }
  const batchName = `IMAGE_${cropIds.length}`;
  subOutputs.push({
    id: stableUuid(`crop-images-${gridSize}x${gridSize}-output-images`),
    name: batchName,
    type: "IMAGE",
    linkIds: [batchOutputLink],
    localized_name: batchName,
    label: "images",
    pos: [1540 + gridSize * 760 + 280, 80 + cropIds.length * 20],
  });
  topOutputs.push({ label: "images", localized_name: batchName, name: batchName, type: "IMAGE", links: [] });

  const subgraph = {
    id: subgraphId,
    version: 1,
    state: { lastGroupId: 1, lastNodeId: nextNodeId - 1, lastLinkId: nextLinkId - 1, lastRerouteId: 0 },
    revision: 0,
    config: {},
    name: `Crop Images ${gridSize}x${gridSize}`,
    inputNode: { id: -10, bounding: [-220, 320, 120, 60] },
    outputNode: { id: -20, bounding: [1540 + gridSize * 760 + 260, 60, 140, Math.max(120, (gridSize * gridSize + 1) * 20 + 40)] },
    inputs: [{
      id: stableUuid(`crop-images-${gridSize}x${gridSize}-input-image`),
      name: "image",
      type: "IMAGE",
      linkIds: inputLinkIds,
      localized_name: "image",
      pos: [-120, 340],
    }],
    outputs: subOutputs,
    widgets: [],
    nodes,
    groups: [{
      id: 1,
      title: `Crop Images ${gridSize}x${gridSize}`,
      bounding: [-260, -80, 1540 + gridSize * 760 + 720, Math.max(1500, gridSize * 560 + 240)],
      color: "#3f789e",
      font_size: 24,
      flags: {},
    }],
    links,
    extra: {},
    category: "Image Tools/Crop",
    description: `Splits an image into a ${gridSize}x${gridSize} grid of ${gridSize * gridSize} tiles.`,
  };

  return {
    revision: 0,
    last_node_id: 1,
    last_link_id: 0,
    nodes: [{
      id: 1,
      type: subgraphId,
      pos: [-2620, 1620],
      size: [230, Math.max(290, 90 + (gridSize * gridSize + 1) * 20)],
      flags: {},
      order: 1,
      mode: 0,
      inputs: [{ localized_name: "image", name: "image", type: "IMAGE", link: null }],
      outputs: topOutputs,
      properties: { "Node name for S&R": subgraphId },
      title: `Crop Images ${gridSize}x${gridSize}`,
    }],
    links: [],
    version: 0.4,
    definitions: { subgraphs: [subgraph] },
    extra: {},
  };
}

fs.mkdirSync(outDir, { recursive: true });
for (const gridSize of [4, 5]) {
  const filename = path.join(outDir, `Crop Images ${gridSize}x${gridSize}.json`);
  fs.writeFileSync(filename, `${JSON.stringify(makeBlueprint(gridSize), null, 2)}\n`);
  console.log(filename);
}
