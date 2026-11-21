// Day of year of the spring NDVI peak observation (qualityMosaic NDVI, April to June), parcel median, 30% subsample
var YILLAR=[2019,2020,2021,2022,2023,2024,2025,2026];
var LISTE=[[0,0.1],[0.1,0.2],[0.2,0.3]];
var parsel = ee.FeatureCollection('projects/ee-tunahancinarr/assets/findik_birlestirilmis').map(function (f) {
  return ee.Feature(f.geometry(), {PID: ee.Number.parse(f.get('system:index'), 16).add(1)});
}).randomColumn('r', 7);
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
YILLAR.forEach(function (y) {
  LISTE.forEach(function (lst, i) {
    var p = parsel.filter(ee.Filter.rangeContains('r', lst[0], lst[1]));
    var b = p.geometry().bounds();
    var pk = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(b).filterDate(ee.Date.fromYMD(y,4,1), ee.Date.fromYMD(y,7,1))
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).linkCollection(csPlus, ['cs_cdf'])
      .map(function (im) { var s = im.select(['B4','B8']).divide(10000).updateMask(im.select('cs_cdf').gte(0.60));
        var doy = ee.Image.constant(ee.Date(im.get('system:time_start')).getRelative('day','year').add(1)).rename('DOY').toFloat();
        return s.normalizedDifference(['B8','B4']).rename('NDVI').addBands(doy.updateMask(s.select('B8').mask())); })
      .qualityMosaic('NDVI').select('DOY');
    var fc = pk.reduceRegions({collection: p, reducer: ee.Reducer.median(), scale: 10, tileScale: 4})
      .filter(ee.Filter.notNull(['median']));
    print(ee.String('Y' + y + 'L' + i + ':').cat(ee.List(fc.toList(5000).map(function (f) {
      f = ee.Feature(f);
      return ee.Number(f.get('PID')).format('%d').cat(',').cat(ee.Number(f.get('median')).round().format('%d'));
    })).join(';')));
  });
});
