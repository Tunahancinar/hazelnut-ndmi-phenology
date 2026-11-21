var YILLAR=[2019,2020];
var LISTE=[[0,0.1],[0.1,0.2],[0.2,0.3]];
var ASSET = 'projects/ee-tunahancinarr/assets/findik_birlestirilmis';
var parsel = ee.FeatureCollection(ASSET).map(function (f) {
  return ee.Feature(f.geometry(), {PID: ee.Number.parse(f.get('system:index'), 16).add(1)});
}).randomColumn('r', 7);
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
function col(bolge, bas, son) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(bolge).filterDate(bas, son)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).linkCollection(csPlus, ['cs_cdf'])
    .map(function (im) { return im.select(['B4','B8','B11']).divide(10000).updateMask(im.select('cs_cdf').gte(0.60)); });
}
YILLAR.forEach(function (y) {
  LISTE.forEach(function (lst, i) {
    var p = parsel.filter(ee.Filter.rangeContains('r', lst[0], lst[1]));
    var b = p.geometry().bounds();
    var mm = col(b, ee.Date.fromYMD(y,5,1), ee.Date.fromYMD(y,7,1)).median();
    var M = mm.normalizedDifference(['B8','B11']).rename('M');
    var pk = col(b, ee.Date.fromYMD(y,4,1), ee.Date.fromYMD(y,7,1))
      .map(function (im) { return im.addBands(im.normalizedDifference(['B8','B4']).rename('NDVI')); })
      .qualityMosaic('NDVI');
    var P = pk.normalizedDifference(['B8','B11']).rename('P');
    var im = M.addBands(P).multiply(10000);
    var fc = im.reduceRegions({collection: p, reducer: ee.Reducer.mean(), scale: 10, tileScale: 4})
      .filter(ee.Filter.notNull(['M','P']));
    print(ee.String('Y' + y + 'L' + i + ':').cat(ee.List(fc.toList(5000).map(function (f) {
      f = ee.Feature(f);
      return ee.Number(f.get('PID')).format('%d').cat(',').cat(ee.Number(f.get('M')).round().format('%d'))
        .cat(',').cat(ee.Number(f.get('P')).round().format('%d'));
    })).join(';')));
  });
});
