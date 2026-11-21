// =====================================================================
// Fındık parselleri (OGM, orman içi/kenarı) için kuraklık modeli veri seti
// Parsel x yıl tablosu, 2019-2026
// Çıktılar Google Drive > FINDIK_GEE klasörüne CSV olarak yazılır.
// =====================================================================

// 1) Asset yolunu kendi yüklediğiniz yol ile değiştirin
var ASSET = 'projects/ee-tunahancinarr/assets/findik_birlestirilmis';
// PID = shapefile satır sırası + 1 (system:index onaltılık satır numarasıdır)
var parsel = ee.FeatureCollection(ASSET).map(function (f) {
  var pid = ee.Number.parse(f.get('system:index'), 16).add(1);
  return ee.Feature(f.geometry(), {PID: pid});
});
var YILLAR = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026];
var KLASOR = 'FINDIK_GEE';
var bolge = parsel.geometry().bounds().buffer(20000);

// ---------------------------------------------------------------------
// Sentinel-2 L2A + Cloud Score+ maskeleme
// ---------------------------------------------------------------------
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
function s2Kompozit(bas, son) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(bolge).filterDate(bas, son)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .linkCollection(csPlus, ['cs_cdf'])
    .map(function (im) {
      var r = im.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
                .divide(10000);
      return r.updateMask(im.select('cs_cdf').gte(0.60));
    });
  var m = col.median();
  var b = function (n) { return m.select(n); };
  var ndvi  = m.normalizedDifference(['B8','B4']).rename('NDVI');
  var evi2  = b('B8').subtract(b('B4')).multiply(2.5)
               .divide(b('B8').add(b('B4').multiply(2.4)).add(1)).rename('EVI2');
  var ndmi  = m.normalizedDifference(['B8','B11']).rename('NDMI');
  var ndre  = m.normalizedDifference(['B8A','B5']).rename('NDRE');
  var gndvi = m.normalizedDifference(['B8','B3']).rename('GNDVI');
  var msi   = b('B11').divide(b('B8')).rename('MSI');
  var cire  = b('B7').divide(b('B5')).subtract(1).rename('CIRE');
  var reswir = b('B5').divide(b('B11')).rename('RE_SWIR');
  var nbr2  = m.normalizedDifference(['B11','B12']).rename('NBR2');
  var n = col.count().select('B8').rename('NOBS');
  return ee.Image.cat([ndvi, evi2, ndmi, ndre, gndvi, msi, cire, reswir, nbr2, n]);
}

// ---------------------------------------------------------------------
// Sentinel-1 GRD (VV, VH, dB ortalaması)
// ---------------------------------------------------------------------
function s1Kompozit(bas, son) {
  var col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(bolge).filterDate(bas, son)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);
  var m = col.mean();
  return m.addBands(m.select('VH').subtract(m.select('VV')).rename('VHVV'));
}

// ---------------------------------------------------------------------
// ERA5-Land aylık: yağış, sıcaklık, VPD, toprak nemi, su dengesi
// ---------------------------------------------------------------------
var era = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR');
function es(tC) { // doygun buhar basıncı (kPa)
  return tC.multiply(17.27).divide(tC.add(237.3)).exp().multiply(0.6108);
}
function eraDonem(y, a1, a2) { // a1..a2 ayları (dahil)
  var c = era.filterDate(ee.Date.fromYMD(y, a1, 1), ee.Date.fromYMD(y, a2, 1).advance(1, 'month'));
  var t   = c.select('temperature_2m').mean().subtract(273.15);
  var tx  = c.select('temperature_2m_max').mean().subtract(273.15);
  var td  = c.select('dewpoint_temperature_2m').mean().subtract(273.15);
  var p   = c.select('total_precipitation_sum').sum().multiply(1000);
  var pet = c.select('potential_evaporation_sum').map(function (i) { return i.abs(); })
             .sum().multiply(1000);
  var sw1 = c.select('volumetric_soil_water_layer_1').mean();
  var sw2 = c.select('volumetric_soil_water_layer_2').mean();
  var sw3 = c.select('volumetric_soil_water_layer_3').mean();
  var vpd = es(t).subtract(es(td)).max(0);
  return {t: t, tx: tx, p: p, pet: pet, sw1: sw1, sw2: sw2, sw3: sw3, vpd: vpd};
}
// 1991-2020 iklim normali (anomali için)
function normal(a1, a2, band, carpan, ekle, toplam) {
  var yillar = ee.List.sequence(1991, 2020);
  var ic = ee.ImageCollection(yillar.map(function (y) {
    var c = era.filterDate(ee.Date.fromYMD(y, a1, 1), ee.Date.fromYMD(y, a2, 1).advance(1, 'month')).select(band);
    var im = toplam ? c.sum() : c.mean();
    return im.multiply(carpan).add(ekle);
  }));
  return {ort: ic.mean(), ss: ic.reduce(ee.Reducer.stdDev())};
}
var nP  = normal(4, 8, 'total_precipitation_sum', 1000, 0, true);   // Nis-Ağu yağış
var nT  = normal(7, 8, 'temperature_2m', 1, -273.15, false);        // Tem-Ağu sıcaklık
var nSW = normal(7, 8, 'volumetric_soil_water_layer_2', 1, 0, false);

// ---------------------------------------------------------------------
// MODIS LST (termal stres)
// ---------------------------------------------------------------------
function lst(bas, son) {
  return ee.ImageCollection('MODIS/061/MOD11A2').filterDate(bas, son)
    .select('LST_Day_1km').mean().multiply(0.02).subtract(273.15).rename('LST_JA');
}

// ---------------------------------------------------------------------
// Yıllık dinamik görüntü
// ---------------------------------------------------------------------
function yilGoruntu(y) {
  var d = function (a, g) { return ee.Date.fromYMD(y, a, g); };
  var am = s2Kompozit(d(4, 1), d(6, 1)).regexpRename('^(.*)$', '$1_AM');
  var ja = s2Kompozit(d(7, 1), d(9, 1)).regexpRename('^(.*)$', '$1_JA');
  var so = s2Kompozit(d(9, 1), d(11, 1)).regexpRename('^(.*)$', '$1_SO');
  var s1am = s1Kompozit(d(4, 1), d(6, 1)).regexpRename('^(.*)$', 'S1_$1_AM');
  var s1ja = s1Kompozit(d(7, 1), d(9, 1)).regexpRename('^(.*)$', 'S1_$1_JA');
  var e1 = eraDonem(y, 4, 8);  // Nisan-Ağustos
  var e2 = eraDonem(y, 7, 8);  // Temmuz-Ağustos
  var e3 = eraDonem(y, 1, 6);  // Ocak-Haziran (ön koşul)
  var iklim = ee.Image.cat([
    e1.p.rename('P_AMJJA'),
    e1.p.subtract(e1.pet).rename('CWB_AMJJA'),
    e3.p.rename('P_JFMAMJ'),
    e2.p.rename('P_JA'),
    e2.t.rename('T_JA'),
    e2.tx.rename('TMAX_JA'),
    e2.vpd.rename('VPD_JA'),
    e2.sw1.rename('SW1_JA'),
    e2.sw2.rename('SW2_JA'),
    e2.sw3.rename('SW3_JA'),
    e1.p.subtract(nP.ort).divide(nP.ss).rename('SPI_AMJJA'),
    e2.t.subtract(nT.ort).rename('TANOM_JA'),
    e2.sw2.subtract(nSW.ort).divide(nSW.ss).rename('SWANOM_JA')
  ]);
  return ee.Image.cat([am, ja, so, s1am, s1ja, iklim, lst(d(7, 1), d(9, 1))]);
}

// ---------------------------------------------------------------------
// Statik değişkenler: topoğrafya + toprak
// ---------------------------------------------------------------------
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').filterBounds(bolge)
  .select('DEM').mosaic().setDefaultProjection('EPSG:4326', null, 30).rename('ELEV');
var terr = ee.Terrain.products(dem);
var slp = terr.select('slope');
var asp = terr.select('aspect').multiply(Math.PI / 180);
var tpi = dem.subtract(dem.focalMean(300, 'circle', 'meters')).rename('TPI300');
var merit = ee.Image('MERIT/Hydro/v1_0_1');
var twi = merit.select('upa').multiply(1e6).divide(90)
  .divide(slp.multiply(Math.PI / 180).tan().max(0.001)).log().rename('TWI');
var topo = ee.Image.cat([dem, slp.rename('SLOPE'), asp.cos().rename('NORTHNESS'),
                         asp.sin().rename('EASTNESS'), tpi, twi]);

function sg(ad, bolen) { // SoilGrids 0-30 ve 30-100 cm kalınlık ağırlıklı ortalama
  var im = ee.Image('projects/soilgrids-isric/' + ad + '_mean').divide(bolen);
  var s = function (k) { return im.select(ad + '_' + k + '_mean'); };
  var ust = s('0-5cm').multiply(5).add(s('5-15cm').multiply(10)).add(s('15-30cm').multiply(15)).divide(30);
  var alt = s('30-60cm').multiply(30).add(s('60-100cm').multiply(40)).divide(70);
  return ust.rename(ad.toUpperCase() + '_0_30').addBands(alt.rename(ad.toUpperCase() + '_30_100'));
}
var toprak = ee.Image.cat([
  sg('clay', 10), sg('sand', 10), sg('silt', 10),   // %
  sg('soc', 10),                                    // g/kg
  sg('bdod', 100),                                  // g/cm3
  sg('cfvo', 10),                                   // % hacim (taşlılık)
  sg('phh2o', 10), sg('cec', 10)
]);
var wc = ee.Image('OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01');
var tk = wc.select('b0').multiply(5).add(wc.select('b10').multiply(15))
  .add(wc.select('b30').multiply(25)).add(wc.select('b60').multiply(35))
  .add(wc.select('b100').multiply(20)).divide(100).rename('FC33_0_100');
var statik = topo.addBands(toprak).addBands(tk);

// ---------------------------------------------------------------------
// Kenar / iç bölge (zararlı-kuraklık ayrımında mekânsal örüntü)
// ---------------------------------------------------------------------
var parcaFC = parsel.map(function (f) {
  var ic = f.geometry().buffer(-20, 1);
  var kenar = f.geometry().difference(ic, 1);
  return ee.FeatureCollection([
    ee.Feature(ic, {PID: f.get('PID'), PARCA: 'IC'}),
    ee.Feature(kenar, {PID: f.get('PID'), PARCA: 'KENAR'})
  ]);
}).flatten().filter(ee.Filter.gt('PID', 0));

// ---------------------------------------------------------------------
// Dışa aktarım
// ---------------------------------------------------------------------
function disa(fc, img, ad, scale) {
  var bandlar = img.bandNames();
  var tablo = img.reduceRegions({collection: fc, reducer: ee.Reducer.mean(),
                                 scale: scale, tileScale: 8})
    .map(function (f) { return f.setGeometry(null); });
  Export.table.toDrive({collection: tablo, description: ad, folder: KLASOR,
                        fileFormat: 'CSV'});
}

disa(parsel, statik, 'FINDIK_STATIK', 30);

YILLAR.forEach(function (y) {
  var im = yilGoruntu(y).set('YIL', y);
  var tablo = im.reduceRegions({collection: parsel, reducer: ee.Reducer.mean(),
                                scale: 10, tileScale: 8})
    .map(function (f) { return f.setGeometry(null).set('YIL', y); });
  Export.table.toDrive({collection: tablo, description: 'FINDIK_DINAMIK_' + y,
                        folder: KLASOR, fileFormat: 'CSV'});
  var ki = s2Kompozit(ee.Date.fromYMD(y, 7, 1), ee.Date.fromYMD(y, 9, 1))
    .select(['NDVI', 'NDMI', 'NDRE', 'GNDVI', 'RE_SWIR']);
  var t2 = ki.reduceRegions({collection: parcaFC, reducer: ee.Reducer.mean(),
                             scale: 10, tileScale: 8})
    .map(function (f) { return f.setGeometry(null).set('YIL', y); });
  Export.table.toDrive({collection: t2, description: 'FINDIK_KENARIC_' + y,
                        folder: KLASOR, fileFormat: 'CSV'});
});

// Hızlı kontrol
var dene = parsel.limit(3);
print(ee.String.encodeJSON(yilGoruntu(2025).reduceRegions(dene, ee.Reducer.mean(), 10).map(function(f){return f.setGeometry(null);})));
print(ee.String.encodeJSON(statik.reduceRegions(dene, ee.Reducer.mean(), 30).map(function(f){return f.setGeometry(null);})));
print(ee.String.encodeJSON(parcaFC.limit(4).map(function(f){return f.setGeometry(null);})));
