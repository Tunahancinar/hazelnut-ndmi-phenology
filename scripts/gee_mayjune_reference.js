var ASSET = 'projects/ee-tunahancinarr/assets/findik_birlestirilmis';
var pts = ee.FeatureCollection(ASSET).map(function (f) {
  return ee.Feature(f.geometry().centroid(10), {PID: ee.Number.parse(f.get('system:index'), 16).add(1)});
});
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
function nd(bolge, bas, son) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(bolge).filterDate(bas, son)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).linkCollection(csPlus, ['cs_cdf'])
    .map(function (im) { return im.select(['B8','B11']).updateMask(im.select('cs_cdf').gte(0.60)); });
  return col.median().normalizedDifference(['B8','B11']);
}
YILLAR.forEach(function (y) {
  PARCA.forEach(function (r) {
    var p = pts.filter(ee.Filter.rangeContains('PID', r[0], r[1]));
    var b = p.geometry().bounds();
    var im = ee.Image.cat([nd(b, ee.Date.fromYMD(y,5,1), ee.Date.fromYMD(y,7,1)).rename('M'),
                           nd(b, ee.Date.fromYMD(y,7,1), ee.Date.fromYMD(y,9,1)).rename('J')])
             .multiply(10000).round().toInt();
    var fc = im.sampleRegions({collection: p, scale: 10, tileScale: 4});
    print(ee.String('Y' + y + 'P' + r[0] + ':').cat(ee.List(fc.toList(20000).map(function (f) {
      f = ee.Feature(f);
      return ee.Number(f.get('PID')).format('%d').cat(',').cat(ee.Number(f.get('M')).format('%d'))
        .cat(',').cat(ee.Number(f.get('J')).format('%d'));
    })).join(';')));
  });
});
