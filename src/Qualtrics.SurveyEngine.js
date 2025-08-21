Qualtrics.SurveyEngine.addOnReady(function() {
    var questionId = this.questionId;
    var labels = jQuery("#" + questionId + " ul.labels");
    labels.hide();
    var firstContainer = jQuery("#" + questionId + " div.slider-container:first");
    jQuery("<div class='label-left'>Model 1</div>").appendTo(firstContainer).css('float', 'left').css('padding-top', '0px');
    jQuery("<div class='label-right'>Model 2</div>").appendTo(firstContainer).css('float', 'right').css('padding-top', '0px');
    var secondContainer = jQuery("#" + questionId + " div.slider-container:last");
    labels.insertAfter(secondContainer);
    labels.show();
  });